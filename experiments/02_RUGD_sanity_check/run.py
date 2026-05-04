"""
Experiment 02: RUGD sanity check (RUGD-native evaluation).

Runs pretrained COCO Panoptic Mask2Former on RUGD images, maps the raw COCO
predictions into RUGD's 25-class space, and reports per-class IoU there.
We deliberately bypass the viplanner color mapping — RUGD GT is the ground
truth, and the natural eval space is RUGD's own 25 classes.

Three learning moments:
  1. label-space alignment   — choose the GT's native space; map predictions to it
  2. NEAREST resampling      — never bilinear-resize a label mask
  3. micro per-class IoU     — accumulate intersection/union, divide once at end
"""

import os
import sys
sys.path.insert(0, "/workspace")

import csv
import cv2
import numpy as np

from mmdet.apis import inference_detector
from viplanner_ros2.m2f_inference import Mask2FormerInference


# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
M2F_DIR    = "/workspace/models/m2f_model/mmdet"
CONFIG     = f"{M2F_DIR}/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py"
CHECKPOINT = f"{M2F_DIR}/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth"

RUGD_ROOT     = "/workspace/data/RUGD"
RUGD_IMAGES   = f"{RUGD_ROOT}/images"
RUGD_ANNOS    = f"{RUGD_ROOT}/annotations"
RUGD_COLORMAP = f"{RUGD_ROOT}/RUGD_annotation-colormap.txt"

OUT_DIR = "/workspace/experiments/02_RUGD_sanity_check/outputs"

IMG_W, IMG_H = 640, 480
INSTANCE_OFFSET = 1000   # MMDet panoptic encoding: pixel = inst_id * 1000 + cat_id
RUGD_VOID_ID = 0


# --------------------------------------------------------------------------- #
# [LEARN 1] Label-space alignment: COCO Panoptic (133) -> RUGD (25)
#
# Hand-written. RUGD_FROM_COCO[rugd_class] = list of COCO class names that
# we consider equivalent. Unlisted COCO classes -> "no RUGD equivalent" (-1).
# Unlisted RUGD classes (mulch, rock-bed, log, picnic-table, bush, ...) just
# get IoU = 0 — that's a real signal of where the pretrained model is blind.
# --------------------------------------------------------------------------- #
RUGD_FROM_COCO = {
    "dirt":     ["dirt-merged"],
    "sand":     ["sand"],
    "grass":    ["grass-merged"],
    "tree":     ["tree-merged", "potted plant", "flower"],
    "sky":      ["sky-other-merged"],
    "water":    ["river", "sea", "water-other"],
    "gravel":   ["gravel"],
    "asphalt":  ["road"],
    "concrete": ["pavement-merged", "platform", "floor-other-merged"],
    "rock":     ["rock-merged", "mountain-merged"],
    "bridge":   ["bridge"],
    "building": ["building-other-merged", "house", "roof"],
    "fence":    ["fence-merged"],
    "vehicle":  ["car", "bus", "truck", "boat", "train", "airplane"],
    "bicycle":  ["bicycle", "motorcycle"],
    "person":   ["person"],
    "pole":     ["fire hydrant", "parking meter"],
    "sign":     ["stop sign", "traffic light"],
}


# --------------------------------------------------------------------------- #
# Helpers — RUGD colormap & lookup tables
# --------------------------------------------------------------------------- #
def load_rugd_colormap(path):
    """Returns (id_to_name, id_to_rgb, name_to_id, rgb_to_id)."""
    id_to_name, id_to_rgb, name_to_id, rgb_to_id = {}, {}, {}, {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            name = parts[1]
            rgb = (int(parts[2]), int(parts[3]), int(parts[4]))
            id_to_name[cid] = name
            id_to_rgb[cid] = rgb
            name_to_id[name] = cid
            rgb_to_id[rgb] = cid
    return id_to_name, id_to_rgb, name_to_id, rgb_to_id


def build_coco_to_rugd_lut(coco_classes, rugd_name_to_id):
    """LUT of length len(coco_classes); -1 means 'no RUGD equivalent'."""
    lut = np.full(len(coco_classes), -1, dtype=np.int16)
    coco_idx = {n: i for i, n in enumerate(coco_classes)}
    for rugd_name, coco_names in RUGD_FROM_COCO.items():
        if rugd_name not in rugd_name_to_id:
            print(f"warn: RUGD class '{rugd_name}' not in colormap")
            continue
        rid = rugd_name_to_id[rugd_name]
        for cn in coco_names:
            if cn not in coco_idx:
                print(f"warn: COCO class '{cn}' not in model class list (skipped)")
                continue
            lut[coco_idx[cn]] = rid
    return lut


def rugd_rgb_to_id(gt_rgb, rugd_rgb_to_id_dict):
    """RUGD GT RGB mask -> per-pixel RUGD class id. -1 for unknown colors."""
    H, W = gt_rgb.shape[:2]
    out = np.full((H, W), -1, dtype=np.int16)
    for rgb, cid in rugd_rgb_to_id_dict.items():
        match = np.all(gt_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
        out[match] = cid
    return out


# --------------------------------------------------------------------------- #
# Inference — bypass Mask2FormerInference.predict(), keep raw COCO ids
# --------------------------------------------------------------------------- #
def predict_coco_id(model, img_bgr):
    """Returns (H, W) int32 array of COCO category indices.
    Pixels with value >= num_classes are 'void' (no class)."""
    result = inference_detector(model, img_bgr)
    sem = result.pred_panoptic_seg.sem_seg.detach().cpu().numpy()[0]   # (H, W)
    return (sem % INSTANCE_OFFSET).astype(np.int32)


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def rugd_id_to_bgr(id_map, rugd_id_to_rgb):
    """Render a RUGD id map to a BGR image. -1 / unknown -> black."""
    H, W = id_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cid, color in rugd_id_to_rgb.items():
        rgb[id_map == cid] = color
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def make_agreement_panel(gt_id, pred_id, valid_mask):
    """Green=correct, red=wrong, gray=ignored (void or unknown GT color)."""
    H, W = gt_id.shape
    panel = np.full((H, W, 3), 128, dtype=np.uint8)
    correct = valid_mask & (gt_id == pred_id)
    wrong   = valid_mask & (gt_id != pred_id)
    panel[correct] = (0, 255, 0)        # green BGR
    panel[wrong]   = (0, 0, 255)        # red BGR
    return panel


def make_legend(present_ids, height, rugd_id_to_name, rugd_id_to_rgb):
    legend_w = 240
    legend = np.ones((height, legend_w, 3), dtype=np.uint8) * 255
    classes = sorted(set(int(c) for c in present_ids if c >= 0))
    if not classes:
        return legend
    row_h = 26
    for i, cid in enumerate(classes):
        if i * row_h + row_h > height:
            break
        y0 = i * row_h
        bgr = tuple(int(x) for x in rugd_id_to_rgb[cid][::-1])  # RGB -> BGR
        cv2.rectangle(legend, (10, y0 + 4), (40, y0 + row_h - 4), bgr, -1)
        cv2.rectangle(legend, (10, y0 + 4), (40, y0 + row_h - 4), (0, 0, 0), 1)
        cv2.putText(legend, f"{cid}: {rugd_id_to_name[cid]}",
                    (50, y0 + row_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return legend


# --------------------------------------------------------------------------- #
# IoU output
# --------------------------------------------------------------------------- #
def save_iou_csv(out_path, rugd_id_to_name, intersection, union):
    iou = intersection / np.maximum(union, 1)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rugd_id", "rugd_class", "intersection", "union", "iou"])
        for cid in range(len(intersection)):
            if union[cid] == 0:
                continue
            w.writerow([cid, rugd_id_to_name.get(cid, "?"),
                        int(intersection[cid]), int(union[cid]),
                        f"{iou[cid]:.4f}"])


def print_iou_summary(rugd_id_to_name, intersection, union):
    iou = intersection / np.maximum(union, 1)
    rows = [(c, intersection[c], union[c], iou[c])
            for c in range(len(intersection)) if union[c] > 0]
    rows.sort(key=lambda r: r[3], reverse=True)
    print("\n=== Per-class IoU (RUGD label space, micro) ===")
    print(f"{'cid':>3} {'class':<26} {'IoU':>7}   {'union(px)':>12}")
    for cid, _i, u, v in rows:
        print(f"{cid:>3} {rugd_id_to_name.get(cid, '?'):<26} {v:>7.4f}   {u:>12d}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Model (we reuse Mask2FormerInference just for its model loading)
    m2f = Mask2FormerInference(config_file=CONFIG, checkpoint_file=CHECKPOINT)
    coco_classes = list(m2f.model.dataset_meta["classes"])
    n_coco = len(coco_classes)
    print(f"COCO model has {n_coco} classes")

    # RUGD label space
    rugd_id_to_name, rugd_id_to_rgb, rugd_name_to_id, rugd_rgb_to_id_dict = \
        load_rugd_colormap(RUGD_COLORMAP)
    n_rugd = max(rugd_id_to_name) + 1   # 25

    # COCO -> RUGD LUT
    coco_to_rugd = build_coco_to_rugd_lut(coco_classes, rugd_name_to_id)

    # Micro-IoU accumulators
    intersection = np.zeros(n_rugd, dtype=np.int64)
    union        = np.zeros(n_rugd, dtype=np.int64)

    for fname in sorted(os.listdir(RUGD_IMAGES)):
        if not fname.endswith(".png"):
            continue

        img_bgr = cv2.imread(os.path.join(RUGD_IMAGES, fname))
        img_bgr = cv2.resize(img_bgr, (IMG_W, IMG_H))

        # [LEARN 2] NEAREST for label masks
        gt_bgr = cv2.imread(os.path.join(RUGD_ANNOS, fname))
        gt_bgr = cv2.resize(gt_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        gt_id  = rugd_rgb_to_id(gt_rgb, rugd_rgb_to_id_dict)

        # Valid pixels: known RUGD color AND not 'void'
        valid = (gt_id >= 0) & (gt_id != RUGD_VOID_ID)

        # Predict in COCO space, remap to RUGD
        coco_id = predict_coco_id(m2f.model, img_bgr)
        if coco_id.shape != (IMG_H, IMG_W):
            coco_id = cv2.resize(coco_id, (IMG_W, IMG_H),
                                 interpolation=cv2.INTER_NEAREST)
        in_range = coco_id < n_coco
        # safe index: clamp out-of-range to 0, then mask back to -1 below
        idx = np.where(in_range, coco_id, 0)
        pred_id = np.where(in_range, coco_to_rugd[idx], -1).astype(np.int16)

        # [LEARN 3] micro IoU: accumulate, divide once at the end
        for cid in range(n_rugd):
            gt_c = (gt_id == cid) & valid
            pr_c = (pred_id == cid) & valid
            intersection[cid] += int(np.sum(gt_c & pr_c))
            union[cid]        += int(np.sum(gt_c | pr_c))

        # Visualization (5 panels)
        gt_vis   = rugd_id_to_bgr(gt_id, rugd_id_to_rgb)
        pred_vis = rugd_id_to_bgr(pred_id, rugd_id_to_rgb)
        agree    = make_agreement_panel(gt_id, pred_id, valid)
        present  = list(np.unique(gt_id)) + list(np.unique(pred_id))
        legend   = make_legend(present, IMG_H, rugd_id_to_name, rugd_id_to_rgb)

        combined = np.hstack([img_bgr, gt_vis, pred_vis, legend])
        cv2.imwrite(os.path.join(OUT_DIR, fname), combined)
        print(f"done: {fname}")

    save_iou_csv(os.path.join(OUT_DIR, "per_class_iou.csv"),
                 rugd_id_to_name, intersection, union)
    print_iou_summary(rugd_id_to_name, intersection, union)
    print(f"\nWrote {OUT_DIR}/per_class_iou.csv")


if __name__ == "__main__":
    main()
