import sys
sys.path.extend(['/home/guozebin/work_code/mast3r', '/home/guozebin/work_code/mast3r/dust3r'])
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import logging
from pathlib import Path
import math
import gradio
from glob import glob
import torch
import argparse
import copy
import numpy as np

from scipy.spatial.transform import Rotation
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess

from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
import open3d as o3d

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def get_camera_intrinsic(w, fov):

    f = fov2focal(math.radians(fov), w)

    return f

def load_best_pairs(pairs_path, imgs_dict, symmetrize=True):
    
    with open(pairs_path, 'r') as f:
        content = f.readlines()

    pairs = [(imgs_dict[Path(i.split()[0]).name], imgs_dict[Path(i.split()[1]).name]) for i in content]

    if symmetrize:
        pairs += [(img2, img1) for img1, img2 in pairs]
        
    return pairs


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-model_path", type=str,
                        default="/home/guozebin/work_code/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth", help="dust3r pretrain model")
    parser.add_argument("-dataset_path", type=str,
                        default="/home/guozebin/work_code/sfm-free-gaussian-splatting/data/demo2", help="dataset")
    parser.add_argument("-lr1", type=int,
                        default=0.01, help="lr")
    parser.add_argument("-lr2", type=int,
                        default=0.005, help="lr")
    parser.add_argument("-niter1", type=int,
                        default=500, help="niter")
    parser.add_argument("-niter2", type=int,
                        default=200, help="niter")
    parser.add_argument("-schedule", type=str,
                        default='cosine', help="[linear, cosine]")
    parser.add_argument("-device", type=str,
                        default='cpu', help="[cpu, cuda]")
    parser.add_argument("-batch", type=int,
                        default=1, help="batch size")
    parser.add_argument("-scene_graph", type=str,
                        default='swin-10', help="[swin, complete, oneref, pairs]")
    parser.add_argument("-optim_level", type=str,
                        default='refine+depth', help='["coarse", "refine", "refine+depth"], Optimization level')
    parser.add_argument("-shared_intrinsics", type=bool,
                        default=False, help="Only optimize one set of intrinsics for all view")
    parser.add_argument("-save_path", type=str,
                        default='output', help="save ckpt path")
    parser.add_argument("-img_fov", type=int,
                        default=90, help="Camera intrinsic parameters")
    parser.add_argument("-matching_conf_thr", type=float,
                        default=5.0, help="Before Fallback to Regr3D!")
    parser.add_argument("-min_conf_thr", type=int,
                        default=2, help="scale ratio for export scale")
    parser.add_argument("-clean_depth", type=bool,
                        default=True, help="scale ratio for export scale")
    parser.add_argument("-focal_adjustment", type=bool,
                        default=False, help="focal adjustment")
    args = parser.parse_args()

    # init parms
    model_path = args.model_path
    dataset_path = args.dataset_path
    device = args.device
    batch_size = args.batch
    schedule = args.schedule
    scene_graph = args.scene_graph
    lr1 = args.lr1
    lr2 = args.lr2
    niter1 = args.niter1
    niter2 = args.niter2
    shared_intrinsics = args.shared_intrinsics
    matching_conf_thr = args.matching_conf_thr
    # 
    img_fov = args.img_fov

    if args.optim_level == 'coarse':
        niter2 = 0
    
    img_fov = args.img_fov
    save_path = str(Path(args.save_path).joinpath(Path(dataset_path).name))
    

    img_fps = sorted(glob(f'{dataset_path}/images/*'))
    # 计算原始的图像wh
    org_h, org_w = cv2.imread(img_fps[0]).shape[:2]

    images, imgs_path = load_images(img_fps, size=512, square_ok=True)
    if len(images) == 1:
        imgs = [images[0], copy.deepcopy(images[0])]
        imgs[1]['idx'] = 1
        img_fps = [img_fps[0], img_fps[0] + '_2']

    # 计算读取之后的图像wh
    h, w = images[0]['img'].shape[-2:]
    best_pairs_path = f"{dataset_path}/pairs-manual.txt"
    if Path(best_pairs_path).exists():
        logging.info('using manual pairs')
        imgs_dict = {Path(k).name:v for k, v in zip(imgs_path, images)}
        pairs = load_best_pairs(pairs_path=best_pairs_path, imgs_dict=imgs_dict)
    else:
        logging.info('using auto pairs')
        pairs = make_pairs(images, scene_graph=scene_graph, symmetrize=True)
        
    model = AsymmetricMASt3R.from_pretrained(model_path).to('cuda')
    # optim
    if img_fov is not None:
        logging.info(f"init focal: {get_camera_intrinsic(w, img_fov)}")
        knowed_focal = get_camera_intrinsic(w, img_fov)
    else:
        knowed_focal = None
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    scene = sparse_global_alignment(img_fps, pairs, os.path.join(save_path, 'cache'),
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in args.optim_level, shared_intrinsics=shared_intrinsics,
                                    knowed_focal=knowed_focal, focal_adjustment=args.focal_adjustment,
                                    matching_conf_thr=matching_conf_thr)

    # export 
    # retrieve useful values from scene:
    imgs = to_numpy(scene.imgs)
    focals = to_numpy(scene.get_focals().cpu())
    cams2world = to_numpy(scene.get_im_poses().cpu())
    # sparse pts3d
    sparse_pts3d = to_numpy(scene.get_sparse_pts3d())
    sparse_pts3d_colors = to_numpy(scene.get_pts3d_colors())
    #dense pts3d
    pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=args.clean_depth))
    confidence_masks = to_numpy([c > args.min_conf_thr for c in confs])

    # export K
    K = to_numpy(scene.get_intrinsics() * (org_w/w))
    K[:, -1, -1] = 1.000
    
    # 保存置信度图和点云图以在进行pnp的时候使用, 以字典的形式构建其表示吧
    pcd_map = {Path(k).name :v for k, v in zip(img_fps, pts3d)}
    pcd_conf = {Path(k).name :v for k, v in zip(img_fps, confidence_masks)} 

    np.save(f'{dataset_path}/pcd_map.npy', pcd_map)
    np.save(f'{dataset_path}/pcd_conf.npy', pcd_conf)


    Ks = {Path(k).name :v for k, v in zip(img_fps, K)}
    c2ws = {Path(k).name :v for k, v in zip(img_fps, cams2world)}

    np.save(f'{dataset_path}/K.npy', Ks)
    np.save(f'{dataset_path}/cams2world.npy', c2ws)

    pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, confidence_masks)])
    col = np.concatenate([p[m] for p, m in zip(imgs, confidence_masks)])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(col)

    o3d.io.write_point_cloud(f"{dataset_path}/points3d_dense.ply", pcd)

    pts = np.concatenate([p for p in sparse_pts3d])
    col = np.concatenate([p for p in sparse_pts3d_colors])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(col)

    o3d.io.write_point_cloud(f"{dataset_path}/points3d_sparse.ply", pcd)