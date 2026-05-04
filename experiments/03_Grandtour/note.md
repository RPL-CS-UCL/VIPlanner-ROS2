# Grandtour
> 这一部分我需要搭建起一个能够引 grandtour dataset (https://grand-tour.leggedrobotics.com/dataset#mission-20-on) 到 viplanner 中来测试后者在现实机器人传感器数据情况下的规划表现能力
目的是能够在实机部署之前就可以通过 grandtour 来检查这个 viplanner 是不是适合部署.

> 任务分成两部分: (1) 首先也是最重要的, 想办法让 viplanner 能够在 grandtour 数据集上运行起来. (2) 其次, 选取一些有价值的场景继续测试 segmentation module 的有效性

> 不过首先我需要你先帮我研究一下 grandtour 怎么搞到本地上来, 以及应该先挑哪些数据集中的场景尝试. 

## Preparation
Download dataset
```bash
klein download \
  --mission d7c37880-a3cf-4aaa-b489-e501591d14b7 \
  --dest /home/data/dataset/grandtour/SNOW-1 --create-dirs --yes \
  "*zed2i_depth*" "*zed2i_images*" "*tf_model*" "*tf_minimal*" "*anymal_state*"
```

```bash
cd /home/data/projects/robohike_ros2_ws/src/RPL-RoboHike-ROS2/src/local_planner/viplanner_ros2
docker build -f docker/Dockerfile.eval-ros -t viplanner-eval-ros . --progress plain

docker run -it --rm --net=host --gpus all \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
  -v /home/data/projects/robohike_ros2_ws/src/RPL-RoboHike-ROS2/src/local_planner/viplanner_ros2:/workspace \
  -v /home/data/dataset/grandtour:/workspace/data/grandtour:ro \
  viplanner-eval-ros /bin/bash

cd /workspace
colcon build --symlink-install --packages-select viplanner_ros2
```