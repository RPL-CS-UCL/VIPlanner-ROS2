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
    └── coco/
        └── panoptic/
            ├── maskformer2_R50_bs16_50ep.yaml   ← mmdet config (m2f_cfg_file param)
            └── model_final_94dc52.pkl           ← Mask2Former weights (m2f_model_path param)
```

## Download Links

### VIPlanner PlannerNet
- Pretrained models: see [leggedrobotics/viplanner README](https://github.com/leggedrobotics/viplanner#model-download)
- Or train your own following [TRAINING.md](../viplanner/TRAINING.md)

### Mask2Former (R50 Panoptic COCO)
- **Config**: from `Mask2Former` repo at `configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml`
  ```bash
  wget -P models/m2f_model/coco/panoptic/ \
    https://raw.githubusercontent.com/facebookresearch/Mask2Former/main/configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml
  ```
- **Weights**: `model_final_94dc52.pkl` from [Mask2Former Model Zoo](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md)
  ```bash
  wget -P models/m2f_model/coco/panoptic/ \
    https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_R50_bs16_50ep/model_final_94dc52.pkl
  ```

## ROS2 Parameter Config (`config/viplanner.yaml`)

```yaml
viplanner_node:
  ros__parameters:
    # VIPlanner model directory (relative to package share or absolute)
    model_save: "models/plannernet"

    # Mask2Former (use absolute paths inside the container)
    m2f_cfg_file:   "/home/developer/ros_ws/src/local_planner/viplanner_ros2/models/m2f_model/coco/panoptic/maskformer2_R50_bs16_50ep.yaml"
    m2f_model_path: "/home/developer/ros_ws/src/local_planner/viplanner_ros2/models/m2f_model/coco/panoptic/model_final_94dc52.pkl"
```
