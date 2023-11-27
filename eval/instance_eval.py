import h5py
import json
import numpy as np
import os
from eval_utils import eval_per_class_ap
from commons import force_mkdir_new

meta = json.load(open("./PartNetE_meta.json", "r"))
save_path = f"/yuchen_slow/partslip_result/result_sam_test"

def gen_gt_labels_partnete(category, model):
    data = np.load(f"/yuchen_fast/partslip_data/test/{category}/{model}/label.npy", allow_pickle=True).item()
    sem_label = data["semantic_seg"]
    ins_label = data["instance_seg"]
    return sem_label, ins_label

def gen_single_shape_gt_h5(category, model):
    sem_label, ins_label = gen_gt_labels_partnete(category, model)
    hf = h5py.File('gt/test-%s.h5' % (model), 'w')
    gt_mask = []
    gt_mask_label = []
    gt_mask_valid = []
    gt_mask_other = np.zeros((1, sem_label.shape[0]), dtype = bool)
    for ins in np.unique(ins_label):
        if ins == -1:
            continue
        idx = np.where(ins_label == ins)[0]
        gt_mask.append(ins_label == ins)
        gt_mask_label.append(sem_label[idx[0]])
        if sem_label[idx[0]] < 0:
            gt_mask_valid.append(False)
            gt_mask_other[0][np.where(ins_label == ins)] = 1
        else:
            gt_mask_valid.append(True)
    gt_mask = np.expand_dims(np.stack(gt_mask), axis=0)
    gt_mask_label = np.array(gt_mask_label, dtype=np.uint8).reshape(1, -1)
    gt_mask_valid = np.array(gt_mask_valid).reshape(1, -1)
    # gt_mask_other = np.zeros((1, sem_label.shape[0]), dtype = bool)
    hf.create_dataset('gt_mask', data=gt_mask)
    hf.create_dataset('gt_mask_label', data=gt_mask_label)
    hf.create_dataset('gt_mask_valid', data=gt_mask_valid)
    hf.create_dataset('gt_mask_other', data=gt_mask_other)
    hf.close()

def load_ply(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
    xyz = []
    rgb = []
    for l in lines[10:]:
        x, y, z = float(l.split(" ")[0]), float(l.split(" ")[1]), float(l.split(" ")[2])
        r, g, b = float(l.split(" ")[3]), float(l.split(" ")[4]), float(l.split(" ")[5])
        xyz.append([x, y, z])
        rgb.append([r * 256 * 256 + g * 256 + b])
    return np.array(xyz), np.array(rgb)

def gen_single_shape_pred_h5(category, model):
    #hf = h5py.File('/media/minghua/data/eval_results/rendering/%s/%s/pred_ins.h5' % (category, model), 'w')
    hf = h5py.File('pred/test-%s.h5' % (model), 'w')
    mask_list = []
    label_list = []
    valid_list = []
    conf_list = []
    pred_dir = "GLIP_pred_8shot_fused/%s" % (", ".join(meta[category]))
    for i, part in enumerate(meta[category]):
        _, label = load_ply(f"{save_path}/%s/%s/instance_seg/%s.ply" % (category, model, part))
        # _, label = load_ply("result_gt_noins_sp+/%s/%s/instance_seg/%s.ply" % (category, model, part))
        if (label != 0).sum() == 0:
            continue
        for ins in np.unique(label):
            if ins == 0:
                continue
            mask_list.append(label == ins)
            label_list.append(i)
            valid_list.append(True)
            conf_list.append(1.0)
    mask_list = np.expand_dims(np.hstack(mask_list), axis=0).transpose((0, 2, 1))
    label_list = np.array(label_list, dtype=np.uint8).reshape(1, -1)
    valid_list = np.array(valid_list).reshape(1, -1)
    conf_list = np.array(conf_list, dtype = np.float32).reshape(1, -1)
    
    hf.create_dataset('mask', data=mask_list)
    hf.create_dataset('label', data=label_list)
    hf.create_dataset('valid', data=valid_list)
    hf.create_dataset('conf', data=conf_list)
    hf.close()

categorys = ["Door"]
for category in categorys:
    force_mkdir_new("gt")
    force_mkdir_new("pred")
    models = os.listdir("/yuchen_fast/partslip_data/test/%s/" % category)
    for model in models:
        print(model)
        try:
            gen_single_shape_gt_h5(category, model)
        except:
            try:
                os.remove(f"gt/test-{model}.h5")
                continue
            except:
                pass
                continue
        try:
            gen_single_shape_pred_h5(category, model)
        except:
            os.remove(f"pred/test-{model}.h5")
            os.remove(f"gt/test-{model}.h5")
    print(eval_per_class_ap(meta[category], "gt", "pred", iou_threshold=0.5))
