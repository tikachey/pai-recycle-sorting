"""Video overlays: captions, picture-in-picture of the AI view, and title cards."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

JP_NAME = {"pet": "ペットボトル", "can": "缶", "glass": "びん", "paper": "紙パック"}


def font(size: int, bold: bool = True):
    names = ("meiryob.ttc", "YuGothB.ttc", "meiryo.ttc") if bold else ("meiryo.ttc", "YuGothM.ttc")
    for name in names + ("NotoSansCJK-Bold.ttc", "NotoSansCJK-Regular.ttc", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


class Hud:
    """Caption bar, badge and picture-in-picture, composited onto each video frame.

    The overlay layer is re-rendered only when the caption or the inset changes.
    """

    def __init__(self, badge: str = "AI: Gemini Robotics-ER 2  |  Sim: MuJoCo  |  1x speed",
                 size=(1280, 720)):
        self._caption = ""
        self._pip: np.ndarray | None = None
        self.badge = badge
        self.size = size
        self.f_cap = font(30)
        self.f_badge = font(18, bold=False)
        self._layer: Image.Image | None = None
        self.labels: list[tuple[str, tuple[float, float], tuple[float, float, float]]] = []
        self.f_label = font(24)

    @property
    def caption(self):
        return self._caption

    @caption.setter
    def caption(self, value: str):
        self._caption = value
        self._layer = None

    @property
    def pip(self):
        return self._pip

    @pip.setter
    def pip(self, value):
        self._pip = value
        self._layer = None

    def _render_layer(self) -> Image.Image:
        w, h = self.size
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        tb = d.textbbox((0, 0), self.badge, font=self.f_badge)
        d.rectangle([12, 12, 12 + tb[2] - tb[0] + 20, 12 + tb[3] - tb[1] + 16], fill=(0, 0, 0, 150))
        d.text((22, 18), self.badge, font=self.f_badge, fill=(255, 255, 255, 255))
        for text, (u, v), rgb in self.labels:
            tb = d.textbbox((0, 0), text, font=self.f_label)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            x0, y0 = u - tw / 2 - 10, v - th / 2 - 8
            d.rounded_rectangle([x0, y0, x0 + tw + 20, y0 + th + 16], radius=8,
                                fill=tuple(int(c * 255) for c in rgb) + (225,))
            d.text((x0 + 10 - tb[0], y0 + 8 - tb[1]), text, font=self.f_label,
                   fill=(255, 255, 255, 255))
        if self._pip is not None:
            pw = 360
            ph = int(pw * self._pip.shape[0] / self._pip.shape[1])
            pip = Image.fromarray(self._pip).convert("RGBA").resize((pw, ph), Image.Resampling.LANCZOS)
            x0, y0 = w - pw - 14, 14
            d.rectangle([x0 - 3, y0 - 3, x0 + pw + 3, y0 + ph + 3], fill=(255, 255, 255, 230))
            layer.paste(pip, (x0, y0))
        if self._caption:
            tb = d.textbbox((0, 0), self._caption, font=self.f_cap)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            y0 = h - th - 46
            d.rectangle([0, y0 - 14, w, h], fill=(0, 0, 0, 165))
            d.text(((w - tw) / 2, y0), self._caption, font=self.f_cap, fill=(255, 255, 255, 255))
        return layer

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self._layer is None:
            self._layer = self._render_layer()
        img = Image.fromarray(frame).convert("RGBA")
        return np.asarray(Image.alpha_composite(img, self._layer).convert("RGB"))


def card(lines: list[tuple[str, int]], size=(1280, 720), bg=(18, 24, 34)) -> np.ndarray:
    """A full-screen text card. lines = [(text, font_size), ...]."""
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    fonts = [font(s) for _, s in lines]
    heights = [d.textbbox((0, 0), t, font=f)[3] - d.textbbox((0, 0), t, font=f)[1]
               for (t, _), f in zip(lines, fonts)]
    gap = 26
    y = (size[1] - sum(heights) - gap * (len(lines) - 1)) / 2
    for (t, _), f, hgt in zip(lines, fonts, heights):
        tb = d.textbbox((0, 0), t, font=f)
        d.text(((size[0] - (tb[2] - tb[0])) / 2 - tb[0], y - tb[1]), t, font=f, fill=(240, 240, 240))
        y += hgt + gap
    return np.asarray(img)


def letterbox(img: np.ndarray, size=(1280, 720), bg=(0, 0, 0)) -> np.ndarray:
    """Fit an image into the video frame without distortion."""
    src = Image.fromarray(img)
    scale = min(size[0] / src.width, size[1] / src.height)
    src = src.resize((int(src.width * scale), int(src.height * scale)), Image.Resampling.LANCZOS)
    out = Image.new("RGB", size, bg)
    out.paste(src, ((size[0] - src.width) // 2, (size[1] - src.height) // 2))
    return np.asarray(out)
