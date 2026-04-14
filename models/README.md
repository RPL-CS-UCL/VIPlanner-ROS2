# VIPlanner Model Files

Place pretrained model files here. **None of these files are tracked by git** (see `.gitignore`).

## Required Structure

```
models/
├── README.md                          ← this file (tracked)
│
├── plannernet/                        ← VIPlanner main model (model_save param)
│   ├── TrainCfg.yaml                  ← training config (auto-generated at training time)
│   └── checkpoints/
│       └── model_final.pth            ← PlannerNet weights
│
└── m2f_model/                         ← Mask2Former (only needed when sem=True)
    └── mmdet/
        ├── mask2former_r50_8xb2-lsj-50e_coco-panoptic.py              ← mmdet config
        └── mask2former_r50_8xb2-lsj-50e_coco-panoptic_*.pth           ← mmdet weights
```

## Download Links

### VIPlanner PlannerNet
- Pretrained models: see [leggedrobotics/viplanner README](https://github.com/leggedrobotics/viplanner#model-download)
- Or train your own following [TRAINING.md](../viplanner/TRAINING.md)

### Mask2Former (R50 Panoptic COCO) — mmdet format

`m2f_inference.py` uses `mmdet.apis.init_detector`, which requires **mmdet-format** files
(a `.py` config + `.pth` checkpoint). Do **not** use the Detectron2-format files from
Facebook's Mask2Former repo (`.yaml` + `.pkl`) — those are incompatible with this API.

Download with `openmim` (install once with `pip install openmim`):

```bash
mim download mmdet \
  --config mask2former_r50_8xb2-lsj-50e_coco-panoptic \
  --dest models/m2f_model/mmdet/
```

This downloads both the `.py` config and the `.pth` checkpoint into `models/m2f_model/mmdet/`.

## ROS2 Parameter Config (`config/viplanner.yaml`)

```yaml
viplanner_node:
  ros__parameters:
    # VIPlanner model directory (absolute path inside container)
    model_save: "/home/developer/ros_ws/src/local_planner/viplanner_ros2/models/plannernet"

    # Mask2Former — mmdet format (use absolute paths inside the container)
    m2f_cfg_file:   "/home/developer/ros_ws/src/local_planner/viplanner_ros2/models/m2f_model/mmdet/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py"
    m2f_model_path: "/home/developer/ros_ws/src/local_planner/viplanner_ros2/models/m2f_model/mmdet/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth"
```
