"""Perception: Gemini Robotics-ER detects and classifies items; depth gives 3D poses."""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .scene import BIN_HALF, CATEGORIES

MODEL = "gemini-robotics-er-2-preview"

DETECT_PROMPT = """You are the perception module of a recycling-sorting robot arm.
The image shows a table with recyclable items, a robot arm, and four colored bins
labeled PET, CAN, GLASS and PAPER.

Detect every recyclable item that is standing on the table surface.
Ignore the robot arm, the bins, the bin labels, and anything already inside a bin.

Return a JSON list. Each element must be:
{"box_2d": [ymin, xmin, ymax, xmax], "label": "<short description>",
 "category": "<pet|can|glass|paper>", "confidence": <0.0-1.0>}

box_2d is normalized to 0-1000. Categories:
- pet: plastic (PET) beverage bottle
- can: aluminum or steel beverage can
- glass: glass bottle
- paper: paper drink carton
"""


@dataclass
class Detection:
    box_2d: list[float]       # [ymin, xmin, ymax, xmax], 0-1000
    label: str
    category: str
    confidence: float = 1.0
    # Filled in by localize()
    x: float | None = None
    y: float | None = None
    yaw: float | None = None
    height: float | None = None
    elongation: float | None = None
    lying: bool = False


# ---------------------------------------------------------------------- camera
def render_rgbd(model, data, camera: str, width: int = 960, height: int = 720):
    with mujoco.Renderer(model, height, width) as r:
        r.update_scene(data, camera=camera)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, camera=camera)
        depth = r.render().copy()
    return rgb, depth


def backproject(model, data, camera: str, depth: np.ndarray) -> np.ndarray:
    """Return an (H, W, 3) array of world coordinates for every pixel."""
    cam = model.camera(camera).id
    h, w = depth.shape
    f = 0.5 * h / np.tan(np.deg2rad(model.cam_fovy[cam]) / 2)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    xc = (u - w / 2) / f * depth
    yc = -(v - h / 2) / f * depth
    zc = -depth
    pc = np.stack([xc, yc, zc], axis=-1)
    R = data.cam_xmat[cam].reshape(3, 3)
    return pc @ R.T + data.cam_xpos[cam]


def project(model, data, camera: str, points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Project world points (N, 3) into pixel coordinates (N, 2) of a camera image."""
    cam = model.camera(camera).id
    f = 0.5 * height / np.tan(np.deg2rad(model.cam_fovy[cam]) / 2)
    pc = (np.asarray(points, float) - data.cam_xpos[cam]) @ data.cam_xmat[cam].reshape(3, 3)
    u = width / 2 + f * pc[:, 0] / -pc[:, 2]
    v = height / 2 - f * pc[:, 1] / -pc[:, 2]
    return np.stack([u, v], axis=1)


# ---------------------------------------------------------------------- Gemini
def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def detect(rgb: np.ndarray, cache_dir: str | Path, prompt: str = DETECT_PROMPT,
           model: str = MODEL, offline: bool = False) -> tuple[list[Detection], str]:
    """Ask Gemini Robotics-ER for item boxes and categories (cached by image hash)."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    png = buf.getvalue()
    key = hashlib.sha256(png + prompt.encode() + model.encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"{key}.json"
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))["response"]
    else:
        if offline:
            raise FileNotFoundError(f"no cached response for this image: {cache}")
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        for attempt in range(6):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=[types.Part.from_bytes(data=png, mime_type="image/png"), prompt],
                    config=types.GenerateContentConfig(temperature=0.2,
                                                       response_mime_type="application/json"),
                )
                break
            except Exception as e:  # free-tier rate limits surface as 429 / 503
                msg = str(e)
                if attempt == 5 or not any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
                    raise
                time.sleep(15 * (attempt + 1))
        raw = resp.text
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"model": model, "response": raw}, ensure_ascii=False,
                                    indent=1), encoding="utf-8")
    dets = []
    for o in _parse_json(raw):
        cat = str(o.get("category", "")).lower()
        if cat not in CATEGORIES or "box_2d" not in o:
            continue
        dets.append(Detection(box_2d=[float(v) for v in o["box_2d"]],
                              label=str(o.get("label", cat)), category=cat,
                              confidence=float(o.get("confidence", 1.0))))
    return dets, raw


# ---------------------------------------------------------------------- 3D pose
def _in_any_bin(x: float, y: float, margin: float = 0.02) -> bool:
    for c in CATEGORIES.values():
        bx, by = c["bin_xy"]
        if abs(x - bx) < BIN_HALF + margin and abs(y - by) < BIN_HALF + margin:
            return True
    return False


# Plausible measured heights [m] per category: (standing, lying on its side).
HEIGHT_RANGE = {
    "pet": ((0.17, 0.23), (0.05, 0.09)),
    "glass": ((0.17, 0.23), (0.05, 0.09)),
    "can": ((0.10, 0.145), (0.05, 0.09)),
    "paper": ((0.08, 0.125), (0.03, 0.07)),
}


def pixel_owners(dets: list[Detection], h: int, w: int) -> np.ndarray:
    """Assign every pixel to at most one detection.

    Where boxes overlap, a pixel belongs to the box whose centre is nearest (in units of
    the box size), so a bottle standing in front of a carton is not mistaken for it.
    Pixels outside every box get -1.
    """
    v, u = np.mgrid[0:h, 0:w]
    u = (u + 0.5) / w * 1000
    v = (v + 0.5) / h * 1000
    best = np.full((h, w), np.inf)
    owner = np.full((h, w), -1, dtype=int)
    for i, det in enumerate(dets):
        y0, x0, y1, x1 = det.box_2d
        inside = (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
        dist = np.abs(u - (x0 + x1) / 2) / max(x1 - x0, 1) + np.abs(v - (y0 + y1) / 2) / max(y1 - y0, 1)
        take = inside & (dist < best)
        best[take] = dist[take]
        owner[take] = i
    return owner


def localize_all(dets: list[Detection], points: np.ndarray) -> list[Detection]:
    """Localize every detection using only its own pixels; drop implausible results."""
    owner = pixel_owners(dets, *points.shape[:2])
    out = []
    for i, det in enumerate(dets):
        loc = localize(det, points, owner == i)
        if loc is None:
            continue
        # Cartons are never flagged as lying, so accept either of their ranges.
        if loc.category == "paper":
            ranges = HEIGHT_RANGE["paper"]
        else:
            ranges = [HEIGHT_RANGE[loc.category][1 if loc.lying else 0]]
        if not any(lo <= loc.height <= hi for lo, hi in ranges):
            continue   # the label and the measured shape disagree: look again next cycle
        out.append(loc)
    return out


def localize(det: Detection, points: np.ndarray, mask: np.ndarray | None = None) -> Detection | None:
    """Estimate the item's position, yaw, height and posture from the depth points in its box.

    The top surface (cap, can lid, carton top) is fully visible from the camera, so the
    centroid of a thin slice below the highest point is an unbiased estimate of the
    item's axis. Its principal axis gives the yaw of cartons and of items lying down.
    """
    h, w, _ = points.shape
    y0, x0, y1, x1 = det.box_2d
    r0, r1 = int(y0 / 1000 * h), int(np.ceil(y1 / 1000 * h))
    c0, c1 = int(x0 / 1000 * w), int(np.ceil(x1 / 1000 * w))
    sl = (slice(max(r0, 0), min(r1, h)), slice(max(c0, 0), min(c1, w)))
    p = points[sl].reshape(-1, 3)
    if mask is not None:
        p = p[mask[sl].reshape(-1)]
    p = p[(p[:, 2] > 0.008) & (p[:, 2] < 0.4)]          # above the table, below the arm
    if len(p) < 30:
        return None
    # Keep the dominant cluster around the median (drops neighbours leaking into the box).
    med = np.median(p[:, :2], axis=0)
    p = p[np.linalg.norm(p[:, :2] - med, axis=1) < 0.06]
    if len(p) < 30:
        return None
    zmax = float(np.percentile(p[:, 2], 98))
    top = p[p[:, 2] > zmax - 0.003][:, :2]
    if len(top) < 10:
        return None
    xy = top.mean(axis=0)
    ridge = p[p[:, 2] > zmax - 0.006][:, :2]
    evals, evecs = np.linalg.eigh(np.cov((ridge - ridge.mean(axis=0)).T))
    major = evecs[:, np.argmax(evals)]
    elong = float(np.sqrt(max(evals) / max(min(evals), 1e-12)))
    if _in_any_bin(*xy):
        return None
    det.x, det.y, det.height = float(xy[0]), float(xy[1]), zmax
    det.yaw = float(np.arctan2(major[1], major[0]))
    det.elongation = elong
    det.lying = det.category != "paper" and (zmax < 0.08 or elong > 3.0)
    return det


# ---------------------------------------------------------------------- drawing
def _font(size: int):
    for name in ("meiryob.ttc", "meiryo.ttc", "YuGothB.ttc", "arialbd.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def annotate(rgb: np.ndarray, dets: list[Detection], title: str | None = None,
             highlight: Detection | None = None) -> np.ndarray:
    img = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(img)
    h, w = rgb.shape[:2]
    font = _font(max(14, h // 40))
    for det in dets:
        y0, x0, y1, x1 = det.box_2d
        box = [x0 / 1000 * w, y0 / 1000 * h, x1 / 1000 * w, y1 / 1000 * h]
        col = tuple(int(v * 255) for v in CATEGORIES[det.category]["rgb"])
        hl = det is highlight
        d.rectangle(box, outline=col, width=8 if hl else 4)
        txt = f"{CATEGORIES[det.category]['label']}  {det.confidence:.2f}" + ("  <- NEXT" if hl else "")
        tb = d.textbbox((box[0], box[1]), txt, font=font)
        ty = box[1] - (tb[3] - tb[1]) - 8
        d.rectangle([box[0], ty, box[0] + tb[2] - tb[0] + 10, box[1]], fill=col)
        d.text((box[0] + 5, ty + 2), txt, font=font, fill=(255, 255, 255))
    if title:
        tf = _font(max(18, h // 28))
        tb = d.textbbox((0, 0), title, font=tf)
        d.rectangle([0, 0, w, tb[3] - tb[1] + 24], fill=(0, 0, 0))
        d.text((16, 10), title, font=tf, fill=(255, 255, 255))
    return np.asarray(img)


def to_dict(dets: list[Detection]) -> list[dict]:
    return [asdict(x) for x in dets]
