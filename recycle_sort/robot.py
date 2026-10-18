"""Panda arm control: damped-least-squares IK and smooth motion primitives."""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
PARK = np.array([0.0, -1.3, 0.0, -2.9, 0.0, 1.75, 0.785])  # out of the AI camera's way
GRIP_OPEN = 255.0
GRIP_CLOSED = 0.0


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def down_rotation(yaw: float) -> np.ndarray:
    """Gripper pointing down (hand z = -world z), rotated by yaw about world z.

    The fingers close along the hand y axis = (sin yaw, -cos yaw, 0).
    """
    c, s = np.cos(yaw), np.sin(yaw)
    x = np.array([c, s, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def yaw_for_closing_dir(phi: float) -> float:
    """Yaw such that the fingers close along world direction angle phi."""
    return phi + np.pi / 2


class Robot:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 on_step: Callable[[], None] | None = None):
        self.m, self.d = model, data
        self.qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.lo = np.array([model.joint(j).range[0] for j in ARM_JOINTS])
        self.hi = np.array([model.joint(j).range[1] for j in ARM_JOINTS])
        self.tcp = model.site("tcp").id
        self.ik_d = mujoco.MjData(model)
        self.on_step = on_step
        self.q_cmd = data.qpos[self.qadr].copy()
        self.grip_cmd = GRIP_OPEN
        self.dt = model.opt.timestep

    # ---------------------------------------------------------------- kinematics
    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.ik_d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.ik_d)
        return self.ik_d.site_xpos[self.tcp].copy(), self.ik_d.site_xmat[self.tcp].reshape(3, 3).copy()

    def _ik(self, pos, rot, q0, iters=300, tol=1e-4):
        q = q0.copy()
        tq = np.zeros(4)
        mujoco.mju_mat2Quat(tq, rot.ravel())
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        cq = np.zeros(4)
        dv = np.zeros(3)
        lam = 0.05
        for _ in range(iters):
            p, r = self.fk(q)
            mujoco.mj_comPos(self.m, self.ik_d)
            mujoco.mju_mat2Quat(cq, r.ravel())
            mujoco.mju_subQuat(dv, tq, cq)
            err = np.concatenate([pos - p, r @ dv])
            if np.linalg.norm(err[:3]) < tol and np.linalg.norm(err[3:]) < 10 * tol:
                break
            mujoco.mj_jacSite(self.m, self.ik_d, jacp, jacr, self.tcp)
            J = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]])
            JJt = J @ J.T + lam**2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            # Null-space pull toward the home posture keeps the elbow tidy.
            Jpinv = J.T @ np.linalg.inv(JJt)
            dq += (np.eye(7) - Jpinv @ J) @ (0.1 * (HOME - q))
            n = np.linalg.norm(dq)
            if n > 0.2:
                dq *= 0.2 / n
            q = np.clip(q + dq, self.lo, self.hi)
        p, r = self.fk(q)
        mujoco.mju_mat2Quat(cq, r.ravel())
        mujoco.mju_subQuat(dv, tq, cq)
        return q, np.linalg.norm(pos - p), np.linalg.norm(dv)

    def ik(self, pos, yaw: float, q0: np.ndarray | None = None, symmetric=True) -> np.ndarray:
        """Solve IK for a top-down grasp. With symmetric=True, yaw and yaw+pi are tried."""
        q0 = self.q_cmd if q0 is None else q0
        best = None
        for y in ([yaw, yaw + np.pi, yaw - np.pi] if symmetric else [yaw]):
            q, ep, er = self._ik(np.asarray(pos, float), down_rotation(y), q0)
            cost = ep * 100 + er + 0.05 * np.linalg.norm(q - q0)
            if best is None or cost < best[0]:
                best = (cost, q, ep, er)
        _, q, ep, er = best
        if ep > 5e-3 or er > 5e-2:
            raise RuntimeError(f"IK failed: pos err {ep:.4f} m, rot err {er:.3f} rad, target {pos}")
        return q

    # ---------------------------------------------------------------- simulation
    def _step(self):
        self.d.ctrl[:7] = self.q_cmd
        self.d.ctrl[7] = self.grip_cmd
        mujoco.mj_step(self.m, self.d)
        if self.on_step:
            self.on_step()

    def wait(self, duration: float):
        for _ in range(int(duration / self.dt)):
            self._step()

    def move_joints(self, q_goal: np.ndarray, duration: float):
        q0 = self.q_cmd.copy()
        n = max(1, int(duration / self.dt))
        for i in range(1, n + 1):
            self.q_cmd = q0 + min_jerk(i / n) * (q_goal - q0)
            self._step()

    def move_to(self, pos, yaw: float, duration: float):
        """Joint-space move to a Cartesian pose (curved path, fast)."""
        self.move_joints(self.ik(pos, yaw), duration)

    def move_linear(self, pos, yaw: float, duration: float, waypoint_dt=0.04):
        """Straight-line Cartesian move with a min-jerk time profile."""
        p0, _ = self.fk(self.q_cmd)
        pos = np.asarray(pos, float)
        # The grasp is symmetric under a half turn: keep the variant closest to now.
        cur = self.current_yaw()
        yaw = min((yaw + k * np.pi for k in range(-3, 4)), key=lambda y: abs(y - cur))
        n_wp = max(2, int(duration / waypoint_dt))
        q_prev = self.q_cmd.copy()
        qs = []
        for k in range(1, n_wp + 1):
            p = p0 + min_jerk(k / n_wp) * (pos - p0)
            q_prev = self.ik(p, yaw, q0=q_prev, symmetric=False)
            qs.append(q_prev)
        steps_per_wp = max(1, int(waypoint_dt / self.dt))
        q_start = self.q_cmd.copy()
        for q_goal in qs:
            for i in range(1, steps_per_wp + 1):
                self.q_cmd = q_start + (i / steps_per_wp) * (q_goal - q_start)
                self._step()
            q_start = q_goal

    def gripper(self, value: float, duration: float = 0.5):
        self.grip_cmd = value
        self.wait(duration)

    def current_yaw(self) -> float:
        _, r = self.fk(self.q_cmd)
        return float(np.arctan2(r[1, 0], r[0, 0]))
