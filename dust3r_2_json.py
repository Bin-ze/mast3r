#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   dust3r_2_json.py
@Time    :   2024/04/28 16:13:22
@Author  :   Bin-ze 
@Version :   1.0
@Desc    :  Description of the script or module goes here. 
将dust3r导出的场景转换为3d gs可以训练的格式：

|-data root
|- |- images
|- |- points3d.ply
|- |- transfroms.json

'''
import os
import numpy as np
import open3d as o3d
import json
import argparse
import logging
import sys
from pathlib import Path
from utils.utils import *

def read_meta(block_path):

    # 读取cams
    block_pose = np.load(block_path.joinpath("cams2world.npy"), allow_pickle=True).item()
    # 读取K
    block_k = np.load(block_path.joinpath("K.npy"), allow_pickle=True).item()
    meta = dict(
        pose=block_pose,
        k=block_k)
    
    return meta

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=str,
                        default='/home/guozebin/work_code/sfm-free-gaussian-splatting/data/demo2', help="source path")
    args = parser.parse_args()

    scene_path = Path(args.s)
    # 读取dust3r格式的pose
    meta = read_meta(scene_path)

    #export 
    write_transformsfile_muti(H=1472, W=1472, Ks=meta['k'], c2ws=meta['pose'], save_path=str(scene_path), trans=None, R=None, suffix=None, scale=1)

    # 
    os.system(f"mv {str(scene_path.joinpath('points3d_sparse.ply'))} {str(scene_path.joinpath('points3d.ply'))}")