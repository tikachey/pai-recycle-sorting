"""Sort items using ground-truth poses (no AI) to validate grasping and recording."""

import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recycle_sort.recorder import Recorder  # noqa: E402
from recycle_sort.robot import PARK, Robot  # noqa: E402
from recycle_sort.scene import CATEGORIES, Item, build_scene, which_bin  # noqa: E402
from recycle_sort.tasks import park, pick_and_place  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "output"


def main():
    items = [
        Item("item0", "pet_bottle", 0.45, 0.10, accent=(0.10, 0.55, 0.25)),
        Item("item1", "can", 0.55, -0.05, accent=(0.80, 0.10, 0.12)),
        Item("item2", "glass_bottle", 0.40, -0.12),
        Item("item3", "paper_carton", 0.58, 0.14, yaw=0.6, accent=(0.15, 0.35, 0.80)),
    ]
    scene = build_scene(items)
    m = scene.model
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    d.qpos[:7] = PARK
    d.ctrl[:7] = PARK
    mujoco.mj_forward(m, d)

    OUT.mkdir(exist_ok=True)
    rec = Recorder(m, d, OUT / "smoke_sort.mp4", camera="video")
    robot = Robot(m, d, on_step=rec.on_step)
    robot.wait(0.5)
    t0 = time.time()
    for it in items:
        body = d.body(it.name)
        x, y = body.xpos[:2]
        w, _, _, z = body.xquat
        yaw = 2 * np.arctan2(z, w)
        pick_and_place(robot, it.type.category, x, y, yaw, it.type.height,
                       CATEGORIES[it.type.category]["bin_xy"])
        robot.wait(0.3)
        print(f"{it.name} ({it.kind}) -> {which_bin(d, m, it.name)} "
              f"(expected {it.type.category})")
    park(robot)
    robot.wait(0.5)
    rec.close()
    print(f"sim {d.time:.1f}s, wall {time.time() - t0:.1f}s, frames {rec.frames}")


if __name__ == "__main__":
    main()
