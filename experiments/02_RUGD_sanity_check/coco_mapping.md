# 现有 COCO → ViPlanner 映射表

来源:
- `_COCO_MAPPING`: [`viplanner_ros2/config/coco_sem_meta.py:146`](../../viplanner_ros2/config/coco_sem_meta.py#L146)
- Loss 常量: [`viplanner_ros2/config/viplanner_sem_meta.py:7-11`](../../viplanner_ros2/config/viplanner_sem_meta.py#L7-L11)

## Loss 含义

| Constant | Value | Meaning |
|---|---|---|
| `TRAVERSABLE_INTENDED_LOSS` | 0.0 | first priority (road, flat terrain etc.) |
| `TRAVERSABLE_UNINTENDED_LOSS` | 0.5 | could walk on but not preferred (gravel/sand/snow) |
| `TERRAIN_LOSS` | 1.0 | natural terrains (grassland/mud land) |
| `ROAD_LOSS` | 1.5 | vehicle road (not good) |
| `OBSTACLE_LOSS` | 2.0 | obstacles |

## Traversable (loss < 2.0)

| ViPlanner Category | Loss | Ground? | COCO Category | Note |
|---|---|---|---|---|
| `sidewalk` | 0.0 | ✓ | `pavement-merged` | pedastrian walk |
| `floor` | 0.0 | ✓ | `floor-other-merged`, `floor-wood`, `platform`, `playingfield`, `rug-merged` | 室内地面 |
| `stairs` | 0.0 | ✓ | `stairs` | 楼梯 |
| `gravel` | 0.5 | ✓ | `gravel` | 沙砾 |
| `sand` | 0.5 | ✓ | `sand` | 沙地 |
| `snow` | 0.5 | ✓ | `snow` | 雪地 |
| `terrain` | 1.0 | ✓ | `grass-merged`, `dirt-merged` | **草地+土地都在这** |
| `indoor_soft` | 1.0 | ✗ | `towel` | 室内软物 (注: ground=False 但 loss=1.0) |
| `road` | 1.5 | ✓ | `road` | 机动车道 |

## 障碍类 (loss = 2.0)

| ViPlanner 类 | Ground? | COCO 来源类 |
|---|---|---|
| `person` | ✗ | `person` |
| `anymal` | ✗ | `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe` |
| `vehicle` | ✗ | `car`, `bus`, `truck`, `boat` |
| `on_rails` | ✗ | `train`, `railroad` |
| `motorcycle` | ✗ | `motorcycle` |
| `bicycle` | ✗ | `bicycle` |
| `building` | ✗ | `building-other-merged`, `house`, `roof` |
| `wall` | ✗ | `wall-other-merged`, `curtain`, `mirror-stuff`, `wall-brick`, `wall-stone`, `wall-tile`, `wall-wood`, `window-blind`, `window-other` |
| `fence` | ✗ | `fence-merged` |
| `bridge` | ✗ | `bridge` |
| `pole` | ✗ | `fire hydrant`, `parking meter` |
| `traffic_sign` | ✗ | `stop sign` |
| `traffic_light` | ✗ | `traffic light` |
| `bench` | ✗ | `bench` |
| `vegetation` | ✗ | `potted plant`, `flower`, `tree-merged`, **`mountain-merged`**, **`rock-merged`** |
| `water_surface` | ✓ | `river`, `sea`, `water-other` (ground=True 但 loss=2.0 — 不希望走但是平面) |
| `sky` | ✗ | `sky-other-merged`, `airplane` |
| `dynamic` | ✗ | 背包/雨伞/手袋/运动器材/餐具/食物/电子设备等 杂项可移动物体 (完整列表见源码) |
| `static` | ✗ | `banner`, `cardboard`, `light`, `tent`, `unknown` |
| `furniture` | ✗ | `chair`, `couch`, `bed`, `dining table`, `toilet`, `clock`, `vase`, `blanket`, `pillow`, `shelf`, `cabinet`, `table-merged`, `counter`, `tv` |
| `door` | ✗ | `door-stuff` |
| `ceiling` | ✗ | `ceiling-merged` |

## ViPlanner 中存在但 COCO 无对应的类

M2F 永远不会输出这些, 但 `VIPLANNER_SEM_META` 里仍定义着:

- `crosswalk` (loss 0.0)
- `tunnel` (loss 2.0)
- `background` (loss 2.0)

---

## 野外场景下值得重点 review 的几条 (初步怀疑)

> 这些只是基于映射表本身 + 你笔记里 "dirt-merged" 线索的推测, 真值要等 RUGD 跑完看 m2f 实际输出来验证.

1. **`dirt-merged` → `terrain` (loss 1.0)**
   你笔记里的例子. 野外土路接近 `gravel` 级别的可通行度, loss 1.0 偏高. 候选改法: 把 `dirt-merged` 单独映射到 `gravel` (0.5).
   *(顺便确认下: 你笔记里写 "原映射表 loss=1.5", 但实际代码里 `terrain = TERRAIN_LOSS = 1.0`. 1.5 是 `road`. )*

2. **`mountain-merged` → `vegetation` (loss 2.0, obstacle)**
   山体被当纯障碍. RUGD 这种山地场景里, 远处缓坡可能是规划目标地形而非要绕开的障碍. 关键是看 m2f 实际会把哪些像素打成 `mountain-merged`.

3. **`rock-merged` → `vegetation` (loss 2.0, obstacle)**
   COCO 不区分小石块铺地 vs 巨石. 野外铺满小石头的路面被一律打成 obstacle 会让规划过度保守 — 这正好对应你笔记里写的 `rock -> rock-bed`.

4. **`grass-merged` → `terrain` (loss 1.0)**
   高草丛 vs 短草坪对腿足机器人难度差异巨大, 单一 loss 粒度太粗 — 不过这是 COCO 标签本身的限制, 映射表层面解不掉, 只能在下游加 cost.

5. **`tree-merged` → `vegetation` (loss 2.0)**
   合理, 但要警惕 m2f 把低矮灌木/丛草归到 `tree-merged` — 那些其实可踩.
