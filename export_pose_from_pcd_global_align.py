#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   export_pose_from_pcd_global_align.py
@Time    :   2024/07/17 09:53:08
@Author  :   Bin-ze 
@Version :   1.0
@Desc    :  Description of the script or module goes here. 
'''

import sys
sys.path.extend(['/home/guozebin/work_code/mast3r', '/home/guozebin/work_code/mast3r/dust3r'])
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import cv2
import torch
import math
import argparse
import logging

import numpy as np
import open3d as o3d

from glob import glob
from pathlib import Path
from dust3r.inference import inference
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.utils import *

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def get_camera_intrinsic(w, fov):

    f = fov2focal(math.radians(fov), w)

    return f

def load_best_pairs(pairs_path, imgs_dict):
    
    with open(pairs_path, 'r') as f:
        content = f.readlines()

    pairs = [(imgs_dict[Path(i.split()[0]).name], imgs_dict[Path(i.split()[1]).name]) for i in content]

    return pairs
    

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-scene_path", type=str,
                        default=None, help="scene_path")
    parser.add_argument("-model_path", type=str,
                        default="/home/guozebin/work_code/mast3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help="dust3r pretrain model")
    parser.add_argument("-dataset_path", type=str,
                        default="/home/guozebin/work_code/dust3r/datasets/zhuanhuayuan/split_block_new/block_3/global", help="dataset")
    parser.add_argument("-lr", type=int,
                        default=0.01, help="lr")
    parser.add_argument("-niter", type=int,
                        default=300, help="niter")
    parser.add_argument("-schedule", type=str,
                        default='cosine', help="[linear, cosine]")
    parser.add_argument("-device", type=str,
                        default='cuda', help="[cpu, cuda]")
    parser.add_argument("-save_interval", type=int,
                        default=300, help="save ckpt interval")
    parser.add_argument("-batch", type=int,
                        default=1, help="batch size")
    parser.add_argument("-scene_graph", type=str,
                        default='swin-10', help="[swin, complete, oneref, pairs]")
    parser.add_argument("-save_path", type=str,
                        default='output', help="save ckpt path")
    parser.add_argument("-img_fov", type=int,
                        default=90, help="Camera intrinsic parameters")
    parser.add_argument("-focal_adjustment", type=bool,
                        default=False, help="focal adjustment")
    parser.add_argument("-init_pose_with_know", type=bool,
                        default=False, help="from know pose start optimizer")
    args = parser.parse_args()

    # init parms
    scene_path = args.scene_path
    model_path = args.model_path
    dataset_path = args.dataset_path
    device = args.device
    batch_size = args.batch
    schedule = args.schedule
    scene_graph = args.scene_graph
    lr = args.lr
    niter = args.niter
    save_iterations = [i for i in range(0, niter, 300)]
    img_fov = args.img_fov
    save_path = str(Path(args.save_path).joinpath(Path(dataset_path).name))
    

    img_fps = sorted(glob(f'{dataset_path}/images/*'))
    # 计算原始的图像wh
    org_h, org_w = cv2.imread(img_fps[0]).shape[:2]

    images, imgs_path = load_images(img_fps, size=512, square_ok=True)

    # 计算读取之后的图像wh
    h, w = images[0]['img'].shape[-2:]
    best_pairs_path = f"{dataset_path}/pairs-manual.txt"
    if Path(best_pairs_path).exists():
        logging.info('using manual pairs')
        imgs_dict = {Path(k).name:v for k, v in zip(imgs_path, images)}
        pairs = load_best_pairs(pairs_path=best_pairs_path, imgs_dict=imgs_dict)
    else:
        logging.info('using auto pairs')
        pairs = make_pairs(images, scene_graph=scene_graph, symmetrize=False)
        
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
    output = inference(pairs=pairs, model=model, device=device, batch_size=batch_size)

    del model
    torch.cuda.empty_cache()
    scene = global_aligner(output, device='cpu', mode=GlobalAlignerMode.PointCloudOptimizer, focal_adjustment=args.focal_adjustment)

    if args.init_pose_with_know:
        # read pose from json
        pose_file = Path(dataset_path).joinpath("transforms.json")
        H, W, K, world_view_transforms, file_names, c2ws = load_pose(pose_file)
        # 姿态初始化正确性检查
        assert [Path(i).name for i in imgs_path] == file_names, "The known image pose must correspond to the input images of dust3r one-to-one!"
        print("befer adjustment")
        scene.preset_pose(known_poses=c2ws)
        pose_adjust_from_know=True

    else:
        pose_adjust_from_know=None

    print("init focal: ",  get_camera_intrinsic(w, img_fov))

    if scene_path is not None:
        scene.load_state_dict(torch.load(scene_path))
        loss = scene.compute_global_alignment(init=None, niter=niter, schedule=schedule, lr=lr, save_iterations=save_iterations, save_path=save_path)
    else:
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr, focal=get_camera_intrinsic(w, img_fov), pose_adjust_from_know=pose_adjust_from_know, save_iterations=save_iterations, save_path=save_path)  

    # export
    if "DUSt3R" in model_path:
        scale = 20
        logging.info(f"rescale scane factor {scale}")
    else:
        scale = 1
        logging.info(f"rescale scane factor {scale}")
    # pose 
    scene = scene.clean_pointcloud()
    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = to_numpy(scene.get_focals())
    cams2world = to_numpy(scene.get_im_poses())
    pts3d = to_numpy(scene.get_pts3d())
    confidence_masks = to_numpy(scene.get_masks())

    if args.init_pose_with_know:
        print("after adjustment ", cams2world[:, :3, 3])

    K = to_numpy(scene.get_intrinsics() * (org_w/w))
    K[:, -1, -1] = 1.000
    
    # 保存置信度图和点云图以在进行pnp的时候使用, 以字典的形式构建其表示吧
    pcd_map = {Path(k).name: v * scale for k, v in zip(img_fps, pts3d)}
    pcd_conf = {Path(k).name: v for k, v in zip(img_fps, confidence_masks)}
    np.save(f'{dataset_path}/pcd_map.npy', pcd_map)
    np.save(f'{dataset_path}/pcd_conf.npy', pcd_conf)

    Ks = {Path(k).name :v for k, v in zip(img_fps, K)}

    cams2world[:, :3, 3] *= scale
    c2ws = {Path(k).name :v for k, v in zip(img_fps, cams2world)}


    np.save(f'{dataset_path}/K.npy', Ks)
    np.save(f'{dataset_path}/cams2world.npy', c2ws)

    pts = np.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)])
    col = np.concatenate([p[m] for p, m in zip(imgs, confidence_masks)])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(col)

    o3d.io.write_point_cloud(f"{dataset_path}/points3d.ply", pcd)


    # # save clean pcd 
    # scene = scene.clean_pointcloud()
    # pts3d_clean = to_numpy(scene.get_pts3d())
    # confidence_masks_clean = to_numpy(scene.get_masks())

    # pts_clean = np.concatenate([p[m] for p, m in zip(pts3d_clean, confidence_masks_clean)])
    # col_clean = np.concatenate([p[m] for p, m in zip(imgs, confidence_masks_clean)])
    # pcd_clean = o3d.geometry.PointCloud()
    # pcd_clean.points = o3d.utility.Vector3dVector(pts_clean)
    # pcd_clean.colors = o3d.utility.Vector3dVector(col_clean)

    # o3d.io.write_point_cloud(f"{dataset_path}/points3d_clean.ply", pcd_clean)
