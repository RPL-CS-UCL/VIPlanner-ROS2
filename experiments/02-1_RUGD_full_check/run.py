"""
Experiment 02-1: RUGD full-check (20% per-scene subset, 18 scenes).

Same label-space alignment + IoU logic as 02_RUGD_sanity_check; differences:
  1. Walks the FULL RUGD dataset under /workspace/data/RUGD_full/
     (mounted from host /home/data/datasets/RUGD/).
  2. Per-scene 20% deterministic stride for IoU, 15 viz-per-scene out of those.
  3. Vectorized confusion-matrix IoU (single np.bincount call).
  4. Per-scene IoU breakdown alongside the overall micro IoU.
  5. Two-row viz with three label spaces side-by-side:
       row 1: [ original | RUGD-GT | RUGD-pred | RUGD-legend ]
       row 2: [ ViPlanner-pred | ViPlanner-legend | COCO-pred | COCO-legend ]
     Lets us eyeball, for the same prediction, what RUGD vs ViPlanner vs raw COCO
     each call it — directly relevant to revising _COCO_MAPPING and RUGD_FROM_COCO.

Note: the COCO->RUGD table (RUGD_FROM_COCO) is FROZEN, copied verbatim from 02.
"""

import os
import sys
sys.path.insert(0, "/workspace")

import csv
import glob
import cv2
import numpy as np
from collections import defaultdict

from mmdet.apis import inference_detector
from tqdm import tqdm

from viplanner_ros2.m2f_inference import Mask2FormerInference
from viplanner_ros2.config.coco_sem_meta import COCO_CATEGORIES, get_class_for_id_mmdet
from viplanner_ros2.config.viplanner_sem_meta import VIPlannerSemMetaHandler


# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
M2F_DIR    = "/workspace/models/m2f_model/mmdet"
CONFIG     = f"{M2F_DIR}/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py"
CHECKPOINT = f"{M2F_DIR}/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth"

RUGD_ROOT     = "/workspace/data/RUGD_full"
RUGD_IMAGES   = f"{RUGD_ROOT}/RUGD_frames-with-annotations"
RUGD_ANNOS    = f"{RUGD_ROOT}/RUGD_annotations"
RUGD_COLORMAP = "/workspace/data/RUGD/RUGD_annotation-colormap.txt"   # stays in-package

OUT_DIR = "/workspace/experiments/02-1_RUGD_full_check/outputs"

IMG_W, IMG_H = 640, 480
INSTANCE_OFFSET = 1000
RUGD_VOID_ID = 0

EVAL_FRACTION = 0.20    # ~1487 of 7436
VIZ_PER_SCENE = 15      # 18 scenes -> ~270 viz PNGs


# --------------------------------------------------------------------------- #
# COCO -> RUGD mapping (frozen, copied from 02)
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
# Helpers — RUGD colormap & lookup tables  (copied from 02)
# --------------------------------------------------------------------------- #
def load_rugd_colormap(path):
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


def build_coco_to_vip_color(coco_classes, vip_meta):
    """LUT (n_coco, 3) RGB; ViPlanner color for each COCO id via _COCO_MAPPING."""
    coco_to_vip_name = get_class_for_id_mmdet(coco_classes)
    out = np.zeros((len(coco_classes), 3), dtype=np.uint8)
    for ci, vname in coco_to_vip_name.items():
        out[ci] = vip_meta.class_color[vname]
    return out


def build_coco_color_table(coco_classes):
    """LUT (n_coco, 3) RGB using the official COCO Panoptic palette."""
    name_to_color = {c["name"]: c["color"] for c in COCO_CATEGORIES}
    out = np.zeros((len(coco_classes), 3), dtype=np.uint8)
    for ci, name in enumerate(coco_classes):
        if name in name_to_color:
            out[ci] = name_to_color[name]
    return out


def rugd_rgb_to_id(gt_rgb, rugd_rgb_to_id_dict):
    H, W = gt_rgb.shape[:2]
    out = np.full((H, W), -1, dtype=np.int16)
    for rgb, cid in rugd_rgb_to_id_dict.items():
        match = np.all(gt_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
        out[match] = cid
    return out


# --------------------------------------------------------------------------- #
# Inference (raw COCO ids; same as 02)
# --------------------------------------------------------------------------- #
def predict_coco_id(model, img_bgr):
    result = inference_detector(model, img_bgr)
    sem = result.pred_panoptic_seg.sem_seg.detach().cpu().numpy()[0]
    return (sem % INSTANCE_OFFSET).astype(np.int32)


# --------------------------------------------------------------------------- #
# Vectorized IoU update (replaces 02's per-class python loop)
# --------------------------------------------------------------------------- #
def update_iou(inter, union, gt_id, pred_id, valid, n_rugd):
    g = gt_id[valid].astype(np.int64)
    p = pred_id[valid].astype(np.int64)
    # Send invalid predictions (-1 or out-of-range) to a dummy bin we discard.
    p_clipped = np.where((p >= 0) & (p < n_rugd), p, n_rugd)
    cm = np.bincount(g * (n_rugd + 1) + p_clipped,
                     minlength=n_rugd * (n_rugd + 1)).reshape(n_rugd, n_rugd + 1)
    cm = cm[:, :n_rugd]                           # drop dummy column
    tp = np.diag(cm)
    gt_total = cm.sum(axis=1)
    pr_total = cm.sum(axis=0)
    inter += tp
    union += gt_total + pr_total - tp


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def rugd_id_to_bgr(id_map, rugd_id_to_rgb):
    H, W = id_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cid, color in rugd_id_to_rgb.items():
        rgb[id_map == cid] = color
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def coco_id_to_vip_bgr(coco_id_map, vip_color_for_coco):
    H, W = coco_id_map.shape
    n_coco = vip_color_for_coco.shape[0]
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    in_range = (coco_id_map >= 0) & (coco_id_map < n_coco)
    rgb[in_range] = vip_color_for_coco[coco_id_map[in_range]]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def coco_id_to_bgr(coco_id_map, coco_color_table):
    H, W = coco_id_map.shape
    n_coco = coco_color_table.shape[0]
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    in_range = (coco_id_map >= 0) & (coco_id_map < n_coco)
    rgb[in_range] = coco_color_table[coco_id_map[in_range]]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


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
        bgr = tuple(int(x) for x in rugd_id_to_rgb[cid][::-1])
        cv2.rectangle(legend, (10, y0 + 4), (40, y0 + row_h - 4), bgr, -1)
        cv2.rectangle(legend, (10, y0 + 4), (40, y0 + row_h - 4), (0, 0, 0), 1)
        cv2.putText(legend, f"{cid}: {rugd_id_to_name[cid]}",
                    (50, y0 + row_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return legend


def make_vip_legend(present_coco_ids, height, coco_to_vip_name, vip_class_color):
    """Legend for the ViPlanner-colored panel.

    Lists the ViPlanner class names that the predicted COCO ids collapsed into.
    Multiple COCO ids -> same ViPlanner class -> shown as one row.
    """
    legend_w = 240
    legend = np.ones((height, legend_w, 3), dtype=np.uint8) * 255
    vip_names = sorted({coco_to_vip_name[int(c)]
                        for c in present_coco_ids
                        if int(c) in coco_to_vip_name})
    if not vip_names:
        return legend
    row_h = 26
    for i, vname in enumerate(vip_names):
        if i * row_h + row_h > height:
            break
        y0 = i * row_h
        rgb = vip_class_color[vname]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.rectangle(legend, (10, y0 + 4), (40, y0 + row_h - 4), bgr, -1)
        cv2.rectangle(legend, (10, y0 + 4), (40, y0 + row_h - 4), (0, 0, 0), 1)
        cv2.putText(legend, vname,
                    (50, y0 + row_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return legend


def make_coco_legend(present_coco_ids, height, coco_classes, coco_color_table):
    """Legend for the raw COCO panel: lists model class names with their COCO colors."""
    legend_w = 240
    legend = np.ones((height, legend_w, 3), dtype=np.uint8) * 255
    n_coco = len(coco_classes)
    ids = sorted({int(c) for c in present_coco_ids if 0 <= int(c) < n_coco})
    if not ids:
        return legend
    row_h = 26
    for i, cid in enumerate(ids):
        if i * row_h + row_h > height:
            break
        y0 = i * row_h
        rgb = coco_color_table[cid]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.rectangle(legend, (10, y0 + 4), (40, y0 + row_h - 4), bgr, -1)
        cv2.rectangle(legend, (10, y0 + 4), (40, y0 + row_h - 4), (0, 0, 0), 1)
        cv2.putText(legend, coco_classes[cid],
                    (50, y0 + row_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return legend


def pad_to_width(panel, target_w):
    h, w = panel.shape[:2]
    if w >= target_w:
        return panel
    pad = np.full((h, target_w - w, 3), 255, dtype=np.uint8)
    return np.hstack([panel, pad])


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def build_subsets(images_root):
    """Returns (eval_paths, viz_set, by_scene_paths_count) — all deterministic."""
    by_scene = defaultdict(list)
    for p in sorted(glob.glob(f"{images_root}/*/*.png")):
        scene = os.path.basename(os.path.dirname(p))
        by_scene[scene].append(p)

    eval_paths, viz_set = [], set()
    for scene, paths in by_scene.items():
        n_eval = max(1, int(round(len(paths) * EVAL_FRACTION)))
        eval_idx = np.linspace(0, len(paths) - 1, n_eval, dtype=int)
        eval_subset = [paths[i] for i in eval_idx]
        eval_paths.extend(eval_subset)

        n_viz = min(VIZ_PER_SCENE, len(eval_subset))
        viz_idx = np.linspace(0, len(eval_subset) - 1, n_viz, dtype=int)
        viz_set.update(eval_subset[i] for i in viz_idx)
    return eval_paths, viz_set, {s: len(p) for s, p in by_scene.items()}


# --------------------------------------------------------------------------- #
# IoU output
# --------------------------------------------------------------------------- #
def save_per_class_csv(out_path, rugd_id_to_name, intersection, union):
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


def save_per_scene_csv(out_path, rugd_id_to_name, per_scene):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "rugd_id", "rugd_class", "intersection", "union", "iou"])
        for scene in sorted(per_scene.keys()):
            inter, union = per_scene[scene]
            iou = inter / np.maximum(union, 1)
            for cid in range(len(inter)):
                if union[cid] == 0:
                    continue
                w.writerow([scene, cid, rugd_id_to_name.get(cid, "?"),
                            int(inter[cid]), int(union[cid]), f"{iou[cid]:.4f}"])


def write_summary(out_path, rugd_id_to_name, total_inter, total_union, per_scene,
                  scene_counts, n_eval_total):
    iou = total_inter / np.maximum(total_union, 1)
    rows = [(c, total_union[c], iou[c]) for c in range(len(total_inter)) if total_union[c] > 0]
    rows.sort(key=lambda r: r[2], reverse=True)

    with open(out_path, "w") as f:
        f.write(f"Evaluated {n_eval_total} images "
                f"(EVAL_FRACTION={EVAL_FRACTION}, {len(per_scene)} scenes)\n\n")
        f.write("=== Overall per-class IoU (RUGD label space, micro) ===\n")
        f.write(f"{'cid':>3} {'class':<26} {'IoU':>7}   {'union(px)':>12}\n")
        for cid, u, v in rows:
            f.write(f"{cid:>3} {rugd_id_to_name.get(cid, '?'):<26} "
                    f"{v:>7.4f}   {u:>12d}\n")

        f.write("\n=== Per-scene mean IoU (averaged over classes with union>0) ===\n")
        f.write(f"{'scene':<14} {'n_total':>7} {'mean_IoU':>9}\n")
        for scene in sorted(per_scene.keys()):
            inter, union = per_scene[scene]
            present = union > 0
            if not present.any():
                continue
            scene_iou = (inter[present] / union[present]).mean()
            f.write(f"{scene:<14} {scene_counts[scene]:>7d} {scene_iou:>9.4f}\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Model
    m2f = Mask2FormerInference(config_file=CONFIG, checkpoint_file=CHECKPOINT)
    coco_classes = list(m2f.model.dataset_meta["classes"])
    n_coco = len(coco_classes)
    print(f"COCO model has {n_coco} classes")

    # RUGD label space
    rugd_id_to_name, rugd_id_to_rgb, rugd_name_to_id, rugd_rgb_to_id_dict = \
        load_rugd_colormap(RUGD_COLORMAP)
    n_rugd = max(rugd_id_to_name) + 1

    # Mapping LUTs
    coco_to_rugd = build_coco_to_rugd_lut(coco_classes, rugd_name_to_id)
    vip_meta = VIPlannerSemMetaHandler()
    vip_color_for_coco = build_coco_to_vip_color(coco_classes, vip_meta)
    coco_to_vip_name = get_class_for_id_mmdet(coco_classes)   # for vip legend
    coco_color_table = build_coco_color_table(coco_classes)   # for raw COCO panel + legend

    # Subsets
    eval_paths, viz_set, scene_counts = build_subsets(RUGD_IMAGES)
    # eval_paths = sorted(glob.glob(f"{RUGD_IMAGES}/creek/*.png"))[:30]
    # viz_set = set(eval_paths[::5])
    # scene_counts = {"creek": 30}

    print(f"Will evaluate {len(eval_paths)} images, "
          f"of which {len(viz_set)} will be visualized.")

    # Accumulators
    total_inter = np.zeros(n_rugd, dtype=np.int64)
    total_union = np.zeros(n_rugd, dtype=np.int64)
    per_scene = {}    # scene -> (inter, union)

    for path in tqdm(eval_paths, desc="full-check"):
        scene = os.path.basename(os.path.dirname(path))
        fname = os.path.basename(path)

        img_bgr = cv2.imread(path)
        img_bgr = cv2.resize(img_bgr, (IMG_W, IMG_H))

        anno_path = os.path.join(RUGD_ANNOS, scene, fname)
        gt_bgr = cv2.imread(anno_path)
        if gt_bgr is None:
            tqdm.write(f"warn: missing annotation for {scene}/{fname}, skipped")
            continue
        gt_bgr = cv2.resize(gt_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        gt_id  = rugd_rgb_to_id(gt_rgb, rugd_rgb_to_id_dict)

        valid = (gt_id >= 0) & (gt_id != RUGD_VOID_ID)

        # COCO inference
        coco_id = predict_coco_id(m2f.model, img_bgr)
        if coco_id.shape != (IMG_H, IMG_W):
            coco_id = cv2.resize(coco_id, (IMG_W, IMG_H),
                                 interpolation=cv2.INTER_NEAREST)
        in_range = coco_id < n_coco
        idx = np.where(in_range, coco_id, 0)
        pred_id = np.where(in_range, coco_to_rugd[idx], -1).astype(np.int16)

        # IoU update — overall + per-scene
        update_iou(total_inter, total_union, gt_id, pred_id, valid, n_rugd)
        if scene not in per_scene:
            per_scene[scene] = (np.zeros(n_rugd, dtype=np.int64),
                                np.zeros(n_rugd, dtype=np.int64))
        s_inter, s_union = per_scene[scene]
        update_iou(s_inter, s_union, gt_id, pred_id, valid, n_rugd)

        # Viz (only for sampled subset)
        if path in viz_set:
            scene_dir = os.path.join(OUT_DIR, scene)
            os.makedirs(scene_dir, exist_ok=True)

            gt_vis      = rugd_id_to_bgr(gt_id, rugd_id_to_rgb)
            pred_vis    = rugd_id_to_bgr(pred_id, rugd_id_to_rgb)
            vip_vis     = coco_id_to_vip_bgr(coco_id, vip_color_for_coco)
            coco_vis    = coco_id_to_bgr(coco_id, coco_color_table)
            present     = list(np.unique(gt_id)) + list(np.unique(pred_id))
            rugd_legend = make_legend(present, IMG_H, rugd_id_to_name, rugd_id_to_rgb)
            present_coco = np.unique(coco_id)
            vip_legend  = make_vip_legend(present_coco, IMG_H,
                                          coco_to_vip_name, vip_meta.class_color)
            coco_legend = make_coco_legend(present_coco, IMG_H,
                                           coco_classes, coco_color_table)

            row1 = np.hstack([img_bgr, gt_vis, pred_vis, rugd_legend])
            row2 = np.hstack([vip_vis, vip_legend, coco_vis, coco_legend])
            target_w = max(row1.shape[1], row2.shape[1])
            combined = np.vstack([pad_to_width(row1, target_w),
                                  pad_to_width(row2, target_w)])
            cv2.imwrite(os.path.join(scene_dir, fname), combined)

    # Outputs
    save_per_class_csv(os.path.join(OUT_DIR, "per_class_iou.csv"),
                       rugd_id_to_name, total_inter, total_union)
    save_per_scene_csv(os.path.join(OUT_DIR, "per_scene_iou.csv"),
                       rugd_id_to_name, per_scene)
    write_summary(os.path.join(OUT_DIR, "summary.txt"),
                  rugd_id_to_name, total_inter, total_union, per_scene,
                  scene_counts, len(eval_paths))
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
