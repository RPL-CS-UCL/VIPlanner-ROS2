import sys
sys.path.insert(0, "/workspace")

import cv2
import os
import torch
import numpy as np

from viplanner_ros2.m2f_inference import Mask2FormerInference
from viplanner_ros2.config.viplanner_sem_meta import VIPLANNER_SEM_META

loss_to_color = {
        0.0: (0, 100, 0),      # 深绿 BGR
        0.5: (0, 200, 100),    # 浅绿
        1.0: (0, 255, 255),    # 黄
        1.5: (0, 165, 255),    # 橙
        2.0: (0, 0, 255),      # 红
    }

def remap_mask(mask):
    H, W = mask.shape[:2]

    loss_map = np.full((H, W), -1.0)

    for class_meta in VIPLANNER_SEM_META:
        rgb = class_meta["color"]
        bgr = (rgb[2], rgb[1], rgb[0])
        matching = np.all(mask == bgr, axis=-1)
        loss_map[matching] = class_meta["loss"]

    loss_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for loss_val, color in loss_to_color.items():
        matching = (loss_map == loss_val)
        loss_mask[matching] = color
    
    return loss_mask

def lookup_present_classes(mask):

    color_to_meta = {
            (c["color"][2], c["color"][1], c["color"][0]): (c["name"], c["loss"]) 
            for c in VIPLANNER_SEM_META
        }    
    unique_colors = np.unique(mask.reshape(-1, 3), axis=0)

    present = []
    for color in unique_colors:
        key = tuple(int(x) for x in color)
        if key in color_to_meta:
            name, loss = color_to_meta[key]
            present.append((key, name, loss))

    return present

def make_dynamic_legend(present_classes, height):
    legend_w = 280
    legend = np.ones((height, legend_w, 3), dtype=np.uint8) * 255  # 白底
    
    if not present_classes:
        return legend
    
    present_classes = sorted(present_classes, key=lambda x: x[2])
    
    n = len(present_classes)
    row_h = 30 
    
    for i, (color_rgb, name, loss) in enumerate(present_classes):
        y0 = i * row_h
                
        cv2.rectangle(legend, (10, y0 + 5), (50, y0 + row_h - 5), color_rgb, -1)
        cv2.rectangle(legend, (10, y0 + 5), (50, y0 + row_h - 5), (0, 0, 0), 1)
        
        text = f"{name} (loss={loss})"
        cv2.putText(legend, text, (60, y0 + row_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    
    return legend

def main():
    
    M2F_DIR = "/workspace/models/m2f_model/mmdet"
    CONFIG = f"{M2F_DIR}/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py"
    CHECKPOINT = f"{M2F_DIR}/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth"

    m2f = Mask2FormerInference(config_file=CONFIG, checkpoint_file=CHECKPOINT)

    input_dir = "/workspace/experiments/01_semantic_sanity_check/inputs"
    output_dir = "/workspace/experiments/01_semantic_sanity_check/outputs"
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(input_dir, fname))
        img = cv2.resize(img, (640,480))

        mask = m2f.predict(img)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

        if mask.shape != img.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        
        present_classes = lookup_present_classes(mask)
        loss_mask = remap_mask(mask)
        overlay = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
        overlay_loss = cv2.addWeighted(img, 0.5, loss_mask, 0.5, 0)
        
        legend = make_dynamic_legend(present_classes, img.shape[0])

        combined = np.hstack([img, mask, overlay, overlay_loss, legend])
        cv2.imwrite(os.path.join(output_dir, fname), combined)

        print(f"done: {fname}")

if __name__ == "__main__":
    main()
