"""Offscreen video recording with optional caption overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import mujoco
import numpy as np


class Recorder:
    """Renders a camera at a fixed frame rate (in simulation time) into an MP4."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, path: str | Path,
                 camera: str = "video", size=(1280, 720), fps: int = 30,
                 overlay: Callable[[np.ndarray], np.ndarray] | None = None):
        self.m, self.d = model, data
        self.camera = camera
        self.fps = fps
        self.overlay = overlay
        self.renderer = mujoco.Renderer(model, size[1], size[0])
        self.writer = imageio.get_writer(str(path), fps=fps, codec="libx264",
                                         quality=8, macro_block_size=1)
        self.next_t = data.time
        self.frames = 0

    def on_step(self):
        if self.d.time + 1e-9 >= self.next_t:
            self.capture()
            self.next_t += 1.0 / self.fps

    def capture(self, repeat: int = 1):
        self.renderer.update_scene(self.d, camera=self.camera)
        frame = self.renderer.render()
        if self.overlay:
            frame = self.overlay(frame)
        for _ in range(repeat):
            self.writer.append_data(frame)
            self.frames += 1

    def hold(self, seconds: float, frame: np.ndarray | None = None):
        """Append a still frame (e.g. a title card or the AI's annotated view)."""
        if frame is None:
            self.capture(repeat=int(seconds * self.fps))
            return
        for _ in range(int(seconds * self.fps)):
            self.writer.append_data(frame)
            self.frames += 1

    def close(self):
        self.writer.close()
        self.renderer.close()
