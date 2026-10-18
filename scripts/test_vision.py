"""One-shot perception test: Gemini detection + depth localization vs. ground truth."""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from recycle_sort.robot import PARK  # noqa: E402
from recycle_sort.scene import Item, build_scene  # noqa: E402
from recycle_sort.vision import annotate, backproject, detect, localize_all, render_rgbd  # noqa: E402

OUT = ROOT / "output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", default=".env")
    args = ap.parse_args()
    load_dotenv(args.env_file)

    items = [
        Item("item0", "pet_bottle", 0.45, 0.10, accent=(0.10, 0.55, 0.25)),
        Item("item1", "can", 0.55, -0.05, accent=(0.80, 0.10, 0.12)),
        Item("item2", "glass_bottle", 0.40, -0.12),
        Item("item3", "paper_carton", 0.58, 0.14, yaw=0.6, accent=(0.15, 0.35, 0.80)),
    ]
    sc = build_scene(items)
    m, d = sc.model, mujoco.MjData(sc.model)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    d.qpos[:7] = PARK
    d.ctrl[:7] = PARK
    for _ in range(300):
        mujoco.mj_step(m, d)

    rgb, depth = render_rgbd(m, d, "ai_cam")
    OUT.mkdir(exist_ok=True)
    Image.fromarray(rgb).save(OUT / "ai_cam.png")
    dets, raw = detect(rgb, OUT / "gemini_cache")
    print("raw:", raw)
    pts = backproject(m, d, "ai_cam", depth)
    located = localize_all(dets, pts)
    for det in located:
        gt = min(items, key=lambda it: np.hypot(d.body(it.name).xpos[0] - det.x,
                                                d.body(it.name).xpos[1] - det.y))
        g = d.body(gt.name).xpos
        err = np.hypot(g[0] - det.x, g[1] - det.y) * 1000
        ok = "OK" if gt.type.category == det.category else "WRONG"
        print(f"{det.category:6s} {ok:5s} '{det.label}' -> ({det.x:.3f},{det.y:.3f}) "
              f"h={det.height:.3f} yaw={np.degrees(det.yaw):6.1f}  "
              f"gt {gt.name}/{gt.kind} err={err:.1f} mm")
    Image.fromarray(annotate(rgb, located, "Gemini Robotics-ER")).save(OUT / "ai_cam_annotated.png")
    (OUT / "detections.json").write_text(json.dumps([d_.__dict__ for d_ in located], indent=1))


if __name__ == "__main__":
    main()
