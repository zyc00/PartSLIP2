import os
import torch
import json
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc
from src.render_pc import render_pc
from src.gen_superpoint import gen_superpoint

def Infer(input_pc_file, category, part_names, zero_shot=False, save_dir="tmp"):
    
    print("[creating tmp dir...]")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    io = IO()
    os.makedirs(save_dir, exist_ok=True)
    
    print("[normalizing input point cloud...]")
    xyz, rgb = normalize_pc(input_pc_file, save_dir, io, device)
    
    print("[rendering input point cloud...]")
    img_dir, pc_idx, screen_coords, num_views = render_pc(xyz, rgb, save_dir, device)
    
    # print('[generating superpoints...]')
    superpoint = gen_superpoint(xyz, rgb, visualize=True, save_dir=save_dir)
    
    print("[finish!]")
    
if __name__ == "__main__":
    partnete_meta = json.load(open("PartNetE_meta.json")) 
    categories = partnete_meta.keys()
    categories_list = [["Box", "Bucket", "Clock", "CoffeeMachine"],
                       ["Dishwasher", "Eyeglasses", "Faucet", "FoldingChair"],
                       ["Lighter", "Microwave", "Mouse", "Pen", "WashingMachine"],
                        ["Phone", "Pliers", "Printer", "Refrigerator", "Window"],
                        ["Remote", "Safe", "Scissors", "Stapler"],
                        ["Switch", "Toilet", "TrashCan", "USB"]]

    # categories = ["Camera", "Cart", "Dispenser", "Kettle"]
    # categories = ["Bottle", "Chair", "Display", "Door"]
    # categories = ["Knife", "Lamp", "StorageFurniture", "Table"]
    # categories = ["KitchenPot", "Oven", "Suitcase", "Toaster"]
    categories = categories_list[5]
    for category in categories:  
        models = os.listdir(f"/yuchen_fast/partslip_data/test/{category}") # list of models
        # models = sorted(models)
        for model in models:
            Infer(f"/yuchen_fast/partslip_data/test/{category}/{model}/pc.ply", category, partnete_meta[category], zero_shot=False, save_dir=f"/yuchen_fast/partslip_data/img_sp/{category}/{model}")
        