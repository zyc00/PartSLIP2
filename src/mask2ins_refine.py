import numpy as np
from utils import save_colored_pc
from scipy.stats import mode
import os
import random
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
from glip_inference import glip_inference, load_model
from utils import get_iou

def load_img(file_name):
    pil_image = Image.open(file_name).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)
    return image

def intersection(lst1, lst2):
    return list(set(lst1).intersection(lst2))

def get_union(f, x): # union-find
    if f[x] == x:
        return x
    f[x] = get_union(f, f[x])
    return f[x]

def calc_sp_connectivity(xyz, superpoints, thr=0.05): 
    # calculate connectivity (bounding box adjacency) between superpoints
    n = len(superpoints)
    X_min, X_max = [], []
    for i in range(n):
        X_min.append(xyz[superpoints[i], :].min(axis=0))
        X_max.append(xyz[superpoints[i], :].max(axis=0))
    X_min = np.array(X_min)
    X_max = np.array(X_max)
    A = (X_min.reshape(n, 1, 3) - X_max.reshape(1, n, 3)).max(axis=2)
    A = np.maximum(A, A.transpose())
    connectivity = A < thr
    return connectivity

def glip_infer(category, save_dir, part_names, num_views, point_idx_all, device, img_dir):
    config ="./GLIP/configs/glip_Swin_L_pt.yaml"
    weight_path = "./models/%s.pth" % category
    glip_demo = load_model(config, weight_path)

    model_type = "vit_h"
    sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda:0")
    sam_predictor = SamPredictor(sam)
    masks_all_view, cat_ids, bboxs = glip_inference(glip_demo, save_dir, img_dir, part_names, sam_predictor, num_views)
    
    pixel_instance_id_all_views = []
    for i in range(num_views):
        valid_mask = (point_idx_all[i] >= 0)
        image = load_img(f"{img_dir}/rendered_img/{i}.png")
        masks = list(masks_all_view[i])
        # masks = sorted(masks, key=lambda x: x.sum(), reverse=False)
        if len(masks) != 0:
            masks = np.stack(masks)
            sort_index = np.argsort(np.array(masks).sum(1).sum(1))
            masks = list(np.array(masks)[sort_index])
            cat_ids[i] = list(np.array(cat_ids[i])[sort_index])
            bboxs[i] = list(np.array(bboxs[i])[sort_index])
            masked = np.zeros_like(masks[0])
            for j, raw_mask in enumerate(masks):
                mask = raw_mask & (~masked)
                masked = mask | masked
            other_mask = valid_mask & (~masked)
        else:
            other_mask = valid_mask
        masks.append(other_mask)
        masks_all_view[i] = masks
        view_cat_ids = list(cat_ids[i])
        view_cat_ids.append(len(part_names))
        cat_ids[i] = view_cat_ids
        masks = np.stack(masks)

        for j, mask in enumerate(masks):
            # visualize
            os.makedirs(f"{save_dir}/sam_result", exist_ok=True)
            masked_image = image * (1 - mask[..., None]) + \
                    mask[..., None] * np.array([255, 0, 0])
            plt.imsave(f"{save_dir}/sam_result/{i}_{j}.jpg", np.uint8(masked_image))

        pixel_instance_id = np.argmax(masks, axis=0).astype(np.int16)
        pixel_instance_id[~valid_mask] = -1
        pixel_instance_id = torch.tensor(pixel_instance_id, device=device)
        pixel_instance_id_all_views.append(pixel_instance_id)
        torch.cuda.empty_cache()
    return pixel_instance_id_all_views, (masks_all_view, cat_ids, bboxs)

def compute_iou(point_logits, point_id):
    num_instances = point_logits.shape[-1]
    iou = np.zeros([num_instances, num_instances])
    point_id_pred = np.argmax(point_logits, -1)
    mask = [point_id == i for i in range(num_instances)]
    mask_pred = [point_id_pred == i for i in range(num_instances)]
    for i in range(num_instances):
        for j in range(num_instances):
            I = np.logical_and(mask[i], mask_pred[j]).sum()
            U = np.logical_or(mask[i], mask_pred[j]).sum()
            iou[i][j] = I / U
    target_ids, pred_ids = linear_sum_assignment(-iou)
    mean_iou = iou[target_ids, pred_ids].mean()
    return mean_iou

def all_pair_bce(pred_prob: torch.Tensor, target_id: torch.Tensor, 
                 valid_mask: torch.Tensor, eps=1e-12):
    ids = torch.arange(target_id.max() + 1, device=target_id.device)
    target_prob = (target_id.unsqueeze(-1) == ids).float()
    target_prob = target_prob.float()
    target_prob = target_prob.unflatten(-1, [-1, 1])
    pred_prob = pred_prob.unflatten(-1, [1, -1])
    positive_ce = -target_prob * torch.log(pred_prob + eps)
    negative_ce = -(1 - target_prob) * torch.log(1 - pred_prob + eps)
    bce = (positive_ce + negative_ce).permute(2, 3, 0, 1) * valid_mask
    return bce.sum([-1, -2])

def load_ply(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
    xyz = []
    rgb = []
    for l in lines[10:]:
        x, y, z = float(l.split(" ")[0]), float(l.split(" ")[1]), float(l.split(" ")[2])
        r, g, b = float(l.split(" ")[3]), float(l.split(" ")[4]), float(l.split(" ")[5])
        xyz.append([x, y, z])
        rgb.append(r * 256 * 256 + g * 256 + b)
    return np.array(xyz), np.array(rgb)

def load_partslip_semantic(category, model, part_names, xyz):
    sem_label = np.zeros([xyz.shape[0]]) - 1
    save_path = f"./result_ps/{category}/{model}/instance_seg"
    for i, part_name in enumerate(part_names):
        _, point_semantic_labels = load_ply(f"{save_path}/{part_name}.ply")
        assert point_semantic_labels.shape[0] == xyz.shape[0]
        for sem in np.unique(point_semantic_labels):
            if sem == 0:
                continue
            sem_label[point_semantic_labels == sem] = i
    return sem_label

def load_partslip(category, model, part_names, xyz, num_instance,
                  num_superpoints, superpoints):
    point_logits = np.zeros([xyz.shape[0], num_instance])
    save_path = f"./result_ps/{category}/{model}/instance_seg"
    instance_idx = 0
    for part_name in part_names:
        _, point_instance_labels = load_ply(f"{save_path}/{part_name}.ply")
        assert point_instance_labels.shape[0] == xyz.shape[0]
        for ins in np.unique(point_instance_labels):
            if ins == 0:
                continue
            point_logits[point_instance_labels == ins, instance_idx] = num_instance
            instance_idx += 1
    sp_logits = np.zeros([num_superpoints, num_instance])
    for i, sp in enumerate(superpoints):
        for p in sp:
            if point_logits[p].sum() != 0:
                sp_logits[i] = point_logits[p]
                break
        
    sp_logits[sp_logits.sum(1) == 0, instance_idx] = num_instance
    # assert (sp_logits.sum(1) == num_instance).all()
    return sp_logits, instance_idx + 1


def sem2ins(xyz, rgb, screen_coor_all, point_idx_all, part_names, 
            save_dir, k_nbrs,
            num_view=10, visualize=True, img_dir=None):
    regenerate_sam = True
    device = torch.device("cuda:0")
    category, model = save_dir.split("/")[-2], save_dir.split("/")[-1]       
    point_instance_id = np.load(f"./data/test/{category}/{model}/label.npy", allow_pickle=True).item()["instance_seg"]
    point_instance_id += 1
    point_instance_id_pad = np.concatenate([point_instance_id, [-1]])
    point_instance_id_pad = torch.as_tensor(point_instance_id_pad, device=device)
    superpoints = np.load(f"./data/img_sp/{category}/{model}/sp.npy", allow_pickle=True)

    medium_save_path = f"./result_ps++/{category}/{model}"
    if not os.path.exists(f"{medium_save_path}/glip_sam_cache_{num_view}.npy") or regenerate_sam:
        pixel_instance_id_all_views, mask_cat_ids = glip_infer(category, save_dir, part_names, num_view,
                                                point_idx_all, device, img_dir)
        torch.save(pixel_instance_id_all_views, f"{save_dir}/glip_sam_cache_{num_view}.npy")
        torch.save(mask_cat_ids, f"{save_dir}/cat_ids_cache_{num_view}.npy")
    else:
        pixel_instance_id_all_views = torch.load(f"{medium_save_path}/glip_sam_cache_{num_view}.npy", map_location=device)
        mask_cat_ids = torch.load(f"{medium_save_path}/cat_ids_cache_{num_view}.npy")

    num_point = point_instance_id.shape[0]
    num_superpoints = superpoints.shape[0]
    point2superpoint = np.zeros(xyz.shape[0])
    for i, sp in enumerate(superpoints):
        for p in sp:
            point2superpoint[p] = i

    print("Start Optimizing...")
    if category == "Keyboard":
        num_instance = 130
    else:
        num_instance = 28
    num_epoch = 10
    H, W = point_idx_all.shape[1:]
    superpoints_logits, pretrained_num_instance = load_partslip(category, model, part_names, xyz, num_instance, num_superpoints, superpoints)
    num_instance = min(pretrained_num_instance + 5, num_instance)
    superpoints_logits = superpoints_logits[:, :num_instance]
    superpoints_logits = torch.tensor(superpoints_logits, device=device).float()
    superpoints_logits.requires_grad = True
    lr = 1
    optimizer = torch.optim.Adam([superpoints_logits], lr)

    torch.cuda.empty_cache()
    masks_all_view, _, _ = mask_cat_ids
    all_masks = [torch.tensor(np.bool_(mask), dtype=torch.bool).cuda().float() for mask in masks_all_view]
    for epoch in range(num_epoch):
        total_cost = 0
        # optimize
        indices = np.random.permutation(num_view)
        for i in range(num_view):
            # define view instance target
            masks = all_masks[indices[i]]
            masks = masks.permute(1, 2, 0)
            point_idx = point_idx_all[indices[i]]
            pixel_instance_id = pixel_instance_id_all_views[indices[i]]
            assert pixel_instance_id.shape == (H, W), pixel_instance_id.shape
            valid_mask = (pixel_instance_id != -1)

            # point logits to pixel
            pixel_logits = superpoints_logits[point2superpoint[point_idx]] * valid_mask[..., None]
            assert pixel_logits.shape == (H, W, num_instance), pixel_logits.shape
            pixel_prob = F.softmax(pixel_logits, dim=-1)

            # get 1 vs all BCE
            cost_torch = all_pair_bce(pixel_prob, pixel_instance_id, valid_mask)
            cost_np = cost_torch.detach().cpu().numpy()
                    
            # match predition and target labels
            target_ids, pred_ids = linear_sum_assignment(cost_np)
            assert len(target_ids) == len(pred_ids)
            cost = cost_torch[target_ids, pred_ids].sum()

            cost.backward()
            total_cost += cost.item()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)      

        print(f"iter: {epoch}, cost: {total_cost / num_view:.2f}")

    # visualize
    masks, cat_ids, bboxs = mask_cat_ids
    instances = []
    superpoint_logits_np = superpoints_logits.detach().cpu().numpy()
    connectivity = calc_sp_connectivity(xyz, superpoints, 0.05)
    for k in range(num_instance):
        ins_sp = np.where(np.argmax(superpoint_logits_np, -1) == k)[0]
        ins = []
        f = []
        for i in range((len(ins_sp))): # initialize union-find sets
            f.append(i)
        for i in range((len(ins_sp))):
            for j in range(i + 1, len(ins_sp)):
                if connectivity[ins_sp[i], ins_sp[j]]:
                    f[get_union(f, i)] = get_union(f, j)
        merged_sps = [[] for i in range(len(ins_sp))]
        for i in range(len(ins_sp)):
            merged_sps[get_union(f, i)].append(superpoints[ins_sp[i]])
        for i in range(len(ins_sp)):
            if len(merged_sps[i]) > 0:
                ins = np.concatenate(merged_sps[i])
                instances.append(ins)

    #filter out instances that have small iou with all bounding boxes
    masks, cat_ids, bboxs = mask_cat_ids
    sem_seg = load_partslip_semantic(category, model, part_names, xyz)
    sem_seg_instance = np.zeros([len(instances)])
    for i, ins in enumerate(instances):
        sem_seg_instance[i] = mode(sem_seg[ins])[0][0]

    flags = [False for _ in range(len(instances))]
    for i in range(num_view):
        screen_coor = screen_coor_all[i] #2D projected location of each 3D point
        visible_pts = np.unique(point_idx_all[i])[1:]
        for k, instance in enumerate(instances):
            if flags[k]:
                continue
            ins_visible_pts = intersection(instance, visible_pts)
            if len(ins_visible_pts) == 0:
                continue
            ins_coor = screen_coor[ins_visible_pts]
            bb1 = {'x1': ins_coor[:, 0].min(), 'y1': ins_coor[:, 1].min(), \
                    'x2': ins_coor[:, 0].max(), 'y2': ins_coor[:, 1].max()}
            for mask, cat_id, bbox in zip(masks[i][:-1], cat_ids[i][:-1], bboxs[i]):
                if cat_id != sem_seg_instance[k]:
                    continue
                x1, y1, x2, y2 = bbox
                bb2 = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                if get_iou(bb1, bb2) > 0.5:
                    flags[k] = True
                    break
    sem_seg_instance = [sem_seg_instance[k] for k in range(len(instances)) if flags[k]]
    instances = [instances[k] for k in range(len(instances)) if flags[k]]
    
    rgb_ins = np.zeros((xyz.shape[0], 3)) 
    
    # save instance segmentation results
    os.makedirs("%s/instance_seg" % save_dir, exist_ok=True)  
    for j in range(len(part_names)):
        rgb_ins = np.zeros((xyz.shape[0], 3)) 
        for i in range(len(instances)):
            if sem_seg_instance[i] == j:
                rgb_ins[instances[i]] = np.random.rand(3)
        save_colored_pc("%s/instance_seg/%s.ply" % (save_dir, part_names[j]), xyz, rgb_ins)
