"""Render still images of the scene from each camera (sanity check)."""

import sys
from pathlib import Path

import mujoco
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recycle_sort.scene import Item, build_scene  # noqa: E402

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
    for _ in range(300):
        mujoco.mj_step(m, d)
    OUT.mkdir(exist_ok=True)
    for cam, (w, h) in {"video": (1280, 720), "video_wide": (1280, 720),
                        "overhead": (960, 720)}.items():
        with mujoco.Renderer(m, h, w) as r:
            r.update_scene(d, camera=cam)
            Image.fromarray(r.render()).save(OUT / f"preview_{cam}.png")
    print("saved to", OUT)


if __name__ == "__main__":
    main()
