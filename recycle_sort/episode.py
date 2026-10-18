"""One sorting episode: observe -> detect -> pick the tallest item -> sort -> repeat."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from .hud import JP_NAME, Hud, card, letterbox
from .recorder import Recorder
from .robot import PARK, Robot
from .scene import BIN_SLOTS, BIN_WALL_H, CATEGORIES, Item, build_scene, which_bin
from .tasks import park, pick_and_place
from .vision import Detection, annotate, backproject, detect, localize_all, project, render_rgbd

CAM = "ai_cam"


class GeminiDetector:
    """Gemini Robotics-ER classifies and boxes the items; depth gives their 3D pose."""
    name = "gemini"

    def __init__(self, cache_dir: Path, offline: bool = False):
        self.cache_dir, self.offline, self.calls = cache_dir, offline, 0

    def __call__(self, m, d, items):
        rgb, depth = render_rgbd(m, d, CAM)
        dets, raw = detect(rgb, self.cache_dir, offline=self.offline)
        self.calls += 1
        pts = backproject(m, d, CAM, depth)
        return rgb, localize_all(dets, pts), raw


class OracleDetector:
    """Evaluation stand-in for the VLM: boxes are ground-truth items projected into the
    camera image. Everything downstream (depth localization, grasping) is unchanged."""
    name = "oracle"

    def __call__(self, m, d, items):
        rgb, depth = render_rgbd(m, d, CAM)
        pts = backproject(m, d, CAM, depth)
        h, w = depth.shape
        dets = []
        for it in items:
            b = m.body(it.name).id
            bp = d.xpos[b]
            if which_bin(d, m, it.name) or bp[2] < -0.1 or bp[2] > 0.06:
                continue      # sorted, fell off the table, or still in the gripper
            corners = []
            for g in np.where((m.geom_bodyid == b) & (m.geom_group == 3))[0]:
                sz = m.geom_size[g]
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = m.geom_dataid[g]
                    local = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
                    ring = np.stack([sz[0] * np.cos(a), sz[0] * np.sin(a)], axis=1)
                    local = np.concatenate([np.c_[ring, np.full(16, s * sz[1])] for s in (-1, 1)])
                else:
                    local = np.array([[sx, sy, sz_] for sx in (-1, 1) for sy in (-1, 1)
                                      for sz_ in (-1, 1)]) * sz[:3]
                corners.append(local @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
            uv = project(m, d, CAM, np.concatenate(corners), w, h)
            u, v = uv[:, 0], uv[:, 1]
            box = [v.min() / h * 1000, u.min() / w * 1000, v.max() / h * 1000, u.max() / w * 1000]
            dets.append(Detection(box_2d=box, label=it.kind, category=it.type.category))
        return rgb, localize_all(dets, pts), "oracle"


def run_episode(items: list[Item], detector, out_dir: Path | None = None, record: bool = False,
                max_cycles: int = 12, verbose: bool = True, trace: bool = False) -> dict:
    scene = build_scene(items)
    m = scene.model
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    d.qpos[:7] = PARK
    d.ctrl[:7] = PARK
    mujoco.mj_forward(m, d)

    hud = rec = None
    if record:
        out_dir.mkdir(parents=True, exist_ok=True)
        hud = Hud()
        # Bin names drawn over the (fixed) video camera image, just behind each bin.
        # Anchors on the table beside each bin, chosen so that no label covers a bin.
        offsets = {"pet": (-0.17, 0.0), "can": (0.17, 0.0), "glass": (0.0, -0.17),
                   "paper": (0.0, -0.17)}
        for key, c in CATEGORIES.items():
            bx, by = c["bin_xy"]
            ox, oy = offsets[key]
            uv = project(m, d, "video", [[bx + ox, by + oy, 0.0]], *hud.size)[0]
            hud.labels.append((JP_NAME[key], tuple(uv), c["rgb"]))
        rec = Recorder(m, d, out_dir / "demo.mp4", camera="video", overlay=hud)
    robot = Robot(m, d, on_step=rec.on_step if rec else None)
    robot.wait(0.4)   # let the items settle

    def caption(text):
        if hud:
            hud.caption = text

    if rec:
        rec.hold(3.5, card([("AI × ロボットで資源ごみを分別する", 52),
                            ("Gemini Robotics-ER 2（認識） × Franka Panda（MuJoCo シミュレーション）", 26),
                            ("動画はすべて 1倍速", 22)]))
        rec.hold(4.0, card([("しくみ", 40),
                            ("① 固定カメラの画像を AI が見て、品目と分別区分を判定", 28),
                            ("② 深度画像から 3D 位置・向き・高さを推定", 28),
                            ("③ アームが 1 個つかんで回収箱へ → ① に戻る（机が空になるまで）", 28)]))
        caption(f"机の上に {len(items)} 個の資源ごみ。どれを、どの箱へ？")
        robot.wait(2.0)

    log = {"items": [it.__dict__ | {"category": it.type.category} for it in items],
           "detector": detector.name, "cycles": []}
    t_start = time.time()
    attempts: list[tuple[float, float]] = []
    dropped = {c: 0 for c in CATEGORIES}   # the robot's own count, used to pick a drop slot
    for cycle in range(1, max_cycles + 1):
        rgb, dets, raw = detector(m, d, items)
        # Give up on an item after two failed attempts at the same spot.
        dets = [x for x in dets
                if sum(np.hypot(x.x - ax, x.y - ay) < 0.04 for ax, ay in attempts) < 2]
        if not dets:
            if rec:
                ann = annotate(rgb, [], f"AIの認識 #{cycle}: 机の上に資源ごみなし → 完了")
                Image.fromarray(ann).save(out_dir / f"cycle{cycle:02d}.png")
                rec.hold(2.5, letterbox(ann))
            log["cycles"].append({"cycle": cycle, "detections": [], "raw": raw})
            break
        # Tallest first, so the hand never sweeps over a taller neighbour;
        # among equals, nearest to the camera first (it may hide the ones behind it).
        target = max(dets, key=lambda x: (round(x.height, 2), x.x))
        attempts.append((target.x, target.y))
        if rec:
            counts = {c: sum(1 for x in dets if x.category == c) for c in CATEGORIES}
            summary = "  ".join(f"{JP_NAME[c]}{n}" for c, n in counts.items() if n)
            ann = annotate(rgb, dets, f"AIの認識 #{cycle}: {len(dets)}個（{summary}）",
                           highlight=target)
            Image.fromarray(ann).save(out_dir / f"cycle{cycle:02d}.png")
            rec.hold(2.2, letterbox(ann))
            hud.pip = ann

        # Ground truth is used only for evaluation/logging, never for control.
        on_table = [it for it in items if which_bin(d, m, it.name) is None]
        gt = min(on_table, key=lambda it: np.hypot(d.body(it.name).xipos[0] - target.x,
                                                   d.body(it.name).xipos[1] - target.y))
        gpos = d.body(gt.name).xipos.copy()   # centre of mass (= axis centre when lying)
        err_mm = float(np.hypot(gpos[0] - target.x, gpos[1] - target.y) * 1000)

        posture = "（倒れている）" if target.lying else ""
        caption(f"#{cycle}  {JP_NAME[target.category]}{posture} →「{JP_NAME[target.category]}」の箱へ")
        status = "ok"

        def tracer(phase, gt=gt):
            if trace:
                p = d.body(gt.name).xpos
                tcp = d.site_xpos[m.site("tcp").id]
                print(f"    {phase:8s} item=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                      f"tcp=({tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f}) finger={d.qpos[7]:.4f}")
        try:
            bx, by = CATEGORIES[target.category]["bin_xy"]
            sx, sy = BIN_SLOTS[dropped[target.category] % len(BIN_SLOTS)]
            pick_and_place(robot, target.category, target.x, target.y, target.yaw,
                           target.height, (bx + sx, by + sy), lying=target.lying,
                           on_phase=tracer)
            dropped[target.category] += 1
        except RuntimeError as e:   # e.g. IK failure
            status = f"skipped: {e}"
        caption("カメラの視界をあけて、もう一度観察")
        park(robot)
        robot.wait(0.4)
        landed = which_bin(d, m, gt.name)
        log["cycles"].append({
            "cycle": cycle, "raw": raw,
            "detections": [dict(x.__dict__) for x in dets],
            "target": {"label": target.label, "category": target.category, "lying": target.lying,
                       "x": target.x, "y": target.y, "yaw": target.yaw, "height": target.height},
            "ground_truth": {"name": gt.name, "kind": gt.kind, "category": gt.type.category,
                             "x": float(gpos[0]), "y": float(gpos[1])},
            "position_error_mm": err_mm, "status": status, "landed_in": landed,
        })
        caption("")
        if hud:
            hud.pip = None
        if verbose:
            print(f"cycle {cycle}: {target.category}{' (lying)' if target.lying else ''} "
                  f"({target.label}) err={err_mm:.1f}mm -> {landed} "
                  f"[gt {gt.type.category}] {status}", flush=True)

    result = {it.name: (it.type.category, which_bin(d, m, it.name)) for it in items}
    correct = sum(1 for cat, landed in result.values() if cat == landed)
    errs = [c["position_error_mm"] for c in log["cycles"] if "position_error_mm" in c]
    log |= {"result": result, "correct": correct, "total": len(items),
            "cycles_used": len(log["cycles"]),
            "detector_calls": getattr(detector, "calls", len(log["cycles"])),
            "mean_position_error_mm": float(np.mean(errs)) if errs else None,
            "sim_time_s": d.time, "wall_time_s": time.time() - t_start}
    if rec:
        rec.hold(4.0, card([("結果", 40),
                            (f"正しい回収箱に入った数： {correct} / {len(items)}", 34),
                            (f"AI 呼び出し {log['detector_calls']} 回 ・ 位置推定の平均誤差 "
                             f"{log['mean_position_error_mm']:.1f} mm", 26)]))
        rec.close()
        log["video_s"] = rec.frames / rec.fps
    return log
