import os
import torch
import json
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc
from src.render_pc import render_pc
from src.glip_inference import glip_inference, load_model
from src.bbox2seg import bbox2seg
from segment_anything import sam_model_registry, SamPredictor

def Infer(input_pc_file, category, model, part_names, zero_shot=False, save_dir="tmp"):
    if zero_shot:
        config ="GLIP/configs/glip_Swin_L.yaml"
        weight_path = "./models/glip_large_model.pth"
        print("-----Zero-shot inference of %s-----" % input_pc_file)
    else:
        config ="GLIP/configs/glip_Swin_L_pt.yaml"
        weight_path = "./models/%s.pth" % category
        print("-----Few-shot inference of %s-----" % input_pc_file)
        
    print("[loading GLIP model...]")
    glip_demo = load_model(config, weight_path)

    print("[creating tmp dir...]")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    io = IO()
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)
    
    print("[normalizing input point cloud...]")
    xyz, rgb = normalize_pc(input_pc_file, save_dir, io, device)
    
    print("[rendering input point cloud...]")
    img_dir, pc_idx, screen_coords, num_views = render_pc(xyz, rgb, save_dir, device)
    
    print("[glip infrence...]")
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "./models/sam_vit_h_4b8939.pth"
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=torch.device("cuda:0"))
    sam_predictor = SamPredictor(sam)
    masks = glip_inference(glip_demo, save_dir, part_names, sam_predictor, num_views=num_views)
    
    print('[generating superpoints...]')
    superpoint = np.load(f"./data/img_sp_more_views/{category}/{model}/sp.npy", allow_pickle=True)
    
    print('[converting bbox to 3D segmentation...]')
    bbox2seg(xyz, superpoint, masks, screen_coords, pc_idx, part_names, save_dir, solve_instance_seg=True, num_view=num_views)
    
    print("[finish!]")
    
if __name__ == "__main__":
    partnete_meta = json.load(open("PartNetE_meta.json")) 
    categories = partnete_meta.keys()
    for category in categories:
        models = os.listdir(f"./data/test/{category}") # list of models
        for model in models:
            Infer(f"./data/test/{category}/{model}/pc.ply", category, model, partnete_meta[category], zero_shot=False, save_dir=f"./result_ps/{category}/{model}")
        