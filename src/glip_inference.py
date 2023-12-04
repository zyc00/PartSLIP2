import os
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

def load_img(file_name):
    pil_image = Image.open(file_name).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def load_model(config_file, weight_file):
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        show_mask_heatmaps=False
    )
    return glip_demo

def draw_rectangle(img, x0, y0, x1, y1):
    color = np.random.rand(3) * 255
    img = img.astype(np.float64)
    img[y0:y1, x0-1:x0+2, :3] = color
    img[y0:y1, x1-1:x1+2, :3] = color
    img[y0-1:y0+2, x0:x1, :3] = color
    img[y1-1:y1+2, x0:x1, :3] = color
    img[y0:y1, x0:x1, :3] /= 2
    img[y0:y1, x0:x1, :3] += color * 0.5
    img = img.astype(np.uint8)
    return img

def save_individual_img(image, bbox, labels, n_cat, pred_dir, view_id):
    n = len(labels)
    result_list = [np.copy(image) for i in range(n_cat)]
    for i in range(n):
        l = labels[i] - 1
        x0, y0, x1, y1 = bbox[i]
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        result_list[l] = draw_rectangle(result_list[l], x0, y0, x1, y1)
    for i in range(n_cat):
        plt.imsave("%s/%d_%d.png" % (pred_dir, view_id, i), result_list[i][:, :, [2, 1, 0]])

def segment(sam_predictor, xyxy) -> np.ndarray:
    masks, scores, logits = sam_predictor.predict(
        box=xyxy,
        multimask_output=True,
    )
    index = np.argmax(scores)
    mask = masks[index]
    return mask

def glip_inference(glip_demo, save_dir, part_names, sam_predictor, num_views=10, 
                    save_pred_img=True, save_individual_img=False, save_pred_json=False):
    pred_dir = os.path.join(save_dir, "glip_pred")
    os.makedirs(pred_dir, exist_ok = True)
    seg_masks = [[] for _ in range(num_views)]
    preds = [[] for _ in range(num_views)]
    for i in range(num_views):
        image = load_img("%s/rendered_img/%d.png" % (save_dir, i))
        result, top_predictions = glip_demo.run_on_web_image(image, part_names, 0.5) 
        if save_pred_img:   
            plt.imsave("%s/%d.png" % (pred_dir, i), result[:, :, [2, 1, 0]])
        bbox = top_predictions.bbox.cpu().numpy()
        score = top_predictions.get_field("scores").cpu().numpy()
        labels = top_predictions.get_field("labels").cpu().numpy()
        if save_individual_img:
            save_individual_img(image, bbox, labels, len(part_names), pred_dir, i)
        for j in range(len(bbox)):
            x1, y1, x2, y2 = bbox[j].tolist()
            preds[i].append((np.array([x1, y1, x2, y2]), labels[j].item() - 1))

    for i in range(num_views):
        image = load_img("%s/rendered_img/%d.png" % (save_dir, i))
        sam_predictor.set_image(image)
        preds_view = preds[i]
        for pred in preds_view:
            bbox, cat_id = pred
            mask = segment(sam_predictor, bbox)
            seg_masks[i].append((mask, cat_id, bbox))
            
    return seg_masks
