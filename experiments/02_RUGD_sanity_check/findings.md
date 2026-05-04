# Experiment 02 — RUGD Sanity Check

Baseline of pretrained COCO Panoptic Mask2Former on RUGD, **before** any fine-tuning.
Evaluated in **RUGD's native 25-class label space** (not viplanner). Predictions
mapped from COCO 133 → RUGD 25 via a hand-written `RUGD_FROM_COCO` table in `run.py`.

## Setup
- Model: Mask2Former R50, COCO Panoptic 50ep checkpoint (same as exp. 01)
- Data: 26 RUGD images at `viplanner_ros2/data/RUGD/{images,annotations}`
- Resolution: resized to 640×480 (NEAREST for GT and prediction)
- Metric: micro per-class IoU in RUGD label space (RUGD `void` ignored)

## How to reproduce
```bash
docker run -it --rm --gpus all \
    -v <viplanner_ros2 repo>:/workspace robohike2 \
    python /workspace/experiments/02_RUGD_sanity_check/run.py
```
Outputs: `outputs/<scene>_<frame>.png` (5-panel) and `outputs/per_class_iou.csv`.

5-panel layout: `[orig | GT (RUGD colors) | pred (RUGD colors) | agreement | legend]`.
Agreement panel: green=correct, red=wrong, gray=ignored (void/unknown).

## Per-class IoU
*(fill in from `outputs/per_class_iou.csv` after the run)*

| RUGD id | class | IoU | union (px) | mapped from COCO | comment |
|---|---|---|---|---|---|
| 1 | dirt | | | dirt-merged | |
| 2 | sand | | | sand | |
| 3 | grass | | | grass-merged | |
| 4 | tree | | | tree-merged + potted plant + flower | |
| 5 | pole | | | fire hydrant + parking meter | weak match |
| 6 | water | | | river + sea + water-other | |
| 7 | sky | | | sky-other-merged | |
| 8 | vehicle | | | car + bus + truck + boat + train + airplane | |
| 9 | container/generic-object | | | (none) | unmapped |
| 10 | asphalt | | | road | |
| 11 | gravel | | | gravel | |
| 12 | building | | | building-other-merged + house + roof | |
| 13 | mulch | | | (none) | unmapped — COCO blind |
| 14 | rock-bed | | | (none) | unmapped — COCO blind |
| 15 | log | | | (none) | unmapped — COCO blind |
| 16 | bicycle | | | bicycle + motorcycle | |
| 17 | person | | | person | |
| 18 | fence | | | fence-merged | |
| 19 | bush | | | (none) | unmapped — COCO has no bush class |
| 20 | sign | | | stop sign + traffic light | |
| 21 | rock | | | rock-merged + mountain-merged | |
| 22 | bridge | | | bridge | |
| 23 | concrete | | | pavement-merged + platform + floor-other-merged | likely confused with asphalt |
| 24 | picnic-table | | | (none) | unmapped |

## Failure modes (top 3)
*(fill in after run)*
1.
2.
3.

## Surprisingly OK (top 3)
1.
2.
3.

## Implications for fine-tuning
- Classes with IoU ≈ 0 that have non-trivial GT support → must be the focus of fine-tune
- Unmapped RUGD classes (mulch, rock-bed, log, bush, picnic-table, container) → COCO model is structurally blind to them; fine-tune is the *only* path
- Asphalt vs concrete — if both have low but non-zero IoU and similar errors, this is a fundamental confusion in the source taxonomy, not a fixable mapping issue
- Classes already > 0.5 → don't disrupt during fine-tune (low loss weight or freeze)

## Notes / gotchas
-
