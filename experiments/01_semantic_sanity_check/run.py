import sys
sys.path.insert(0, "/workspace")

import cv2
import os
import torch

from viplanner_ros2.m2f_inference import Mask2FormerInference

M2F_DIR = "/workspace/models/m2f_model/mmdet"
CONFIG = f"{M2F_DIR}/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py"
CHECKPOINT = f"{M2F_DIR}/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth"

m2f = Mask2FormerInference(config_file=CONFIG, checkpoint_file=CHECKPOINT)

import cv2

input_dir = "/workspace/experiments/01_semantic_sanity_check/inputs"
output_dir = "/workspace/experiments/01_semantic_sanity_check/outputs"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.endswith(".jpg"):
        continue
    img = cv2.imread(os.path.join(input_dir, fname))
    img = cv2.resize(img, (640,480))
    mask = m2f.predict(img)
    cv2.imwrite(os.path.join(output_dir, fname), mask)
    print(f"done: {fname}")