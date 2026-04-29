import matplotlib.pyplot as plt
import numpy as np

levels = [
    ("cost 0.0  -  intended (sidewalk, floor, stairs, crosswalk)",  (0.0, 0.8, 0.0)),
    ("cost 0.5  -  unintended (gravel, sand, snow)",                (0.6, 0.8, 0.0)),
    ("cost 1.0  -  terrain (grass, dirt)",                          (1.0, 1.0, 0.0)),
    ("cost 1.5  -  road",                                           (1.0, 0.5, 0.0)),
    ("cost 2.0  -  obstacle (everything else)",                     (0.8, 0.0, 0.0)),
]

fig, axes = plt.subplots(len(levels), 1, figsize=(8,3))

for ax, (label, color) in zip (axes, levels):
    ax.imshow(np.full((1, 10, 3), color))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(label, loc="left", fontsize=11)

plt.tight_layout()
plt.savefig("legend.png", dpi=150, bbox_inches="tight")
print("saved legend.png")