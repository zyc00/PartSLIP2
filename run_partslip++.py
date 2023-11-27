from src.mask2ins_refine import sem2ins
from pytorch3d.io import IO
import os
from utils import normalize_pc
import torch
import numpy as np
import json

def test(input_pc_file, part_names, sp_dir, save_dir="tmp"):
    io = IO()
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda:0")

    xyz, rgb = normalize_pc(input_pc_file, save_dir, io, device)

    idx_dir = f"{sp_dir}/idx_dir"
    pc_idx = np.load(f"{sp_dir}/idx.npy", allow_pickle=True)
    screen_coords = np.load(f"{sp_dir}/coor.npy", allow_pickle=True)

    sem2ins(xyz, rgb, screen_coords, pc_idx, part_names, 
                   save_dir, 20, pc_idx.shape[0], img_dir=sp_dir)
    
if __name__ == "__main__":
    partnete_meta = json.load(open("PartNetE_meta.json")) 
    categories = partnete_meta.keys()

    for category in categories:
        models = os.listdir(f"./data/img_sp/{category}") # list of models
        for model in models:
            print(f"Category: {category}, Model: {model}")
            test(f"./data/test/{category}/{model}/pc.ply", partnete_meta[category], 
                 sp_dir=f"./data/img_sp/{category}/{model}",
                 save_dir=f"./result_ps++/{category}/{model}")