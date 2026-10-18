"""Task-level motions: pick an item and drop it into a bin."""

from __future__ import annotations

import numpy as np

from .robot import GRIP_CLOSED, GRIP_OPEN, PARK, Robot
from .scene import BIN_WALL_H

CLEAR_Z = 0.25        # item bottom height while carrying (clears standing bottles)
RELEASE_Z = 0.13      # item bottom height when released above a bin
SAFE_Z = 0.30         # TCP height for travelling above the items
PRE_GRASP_DZ = 0.10   # minimum approach distance above the grasp point

# How far below the item's top the fingers grasp, per category [m].
# Standing bottles are held by the cap/neck, cans and cartons near the top rim;
# items lying down are held around their middle.
GRASP_DEPTH = {"pet": 0.014, "glass": 0.022, "can": 0.027, "paper": 0.028}
LYING_GRASP_DEPTH = 0.030


def closing_direction(category: str, lying: bool, x: float, y: float, item_yaw: float,
                      bin_xy: tuple[float, float]) -> float:
    """World angle along which the fingers should close."""
    if category == "paper" or lying:
        # Across the narrow side: perpendicular to the item's long axis.
        return item_yaw + np.pi / 2
    # Round items standing up are held by the cap/neck and can swing like a pendulum
    # about the line through the two finger pads. Aligning that line with the
    # transport direction means the arm's acceleration produces no swinging torque.
    return float(np.arctan2(bin_xy[1] - y, bin_xy[0] - x))


def pick_and_place(robot: Robot, category: str, x: float, y: float, item_yaw: float,
                   top_z: float, bin_xy: tuple[float, float], lying: bool = False,
                   speed: float = 1.0, on_phase=None) -> None:
    """Pick an item whose top is at top_z and drop it into the bin at bin_xy."""
    phase = on_phase or (lambda name: None)
    phi = closing_direction(category, lying, x, y, item_yaw, bin_xy)
    yaw = phi + np.pi / 2            # hand yaw whose y axis (finger travel) is phi
    gz = top_z - (LYING_GRASP_DEPTH if lying else GRASP_DEPTH[category])
    gz = max(gz, 0.02)
    s = 1.0 / speed
    robot.grip_cmd = GRIP_OPEN
    phase("approach")
    robot.move_to([x, y, max(gz + PRE_GRASP_DZ, SAFE_Z)], yaw, 1.6 * s)
    robot.move_linear([x, y, gz], yaw, 1.1 * s)
    phase("grasp")
    robot.gripper(GRIP_CLOSED, 0.7)
    carry = CLEAR_Z + gz
    phase("lift")
    robot.move_linear([x, y, carry], yaw, 0.9 * s)
    phase("carry")
    bx, by = bin_xy
    robot.move_linear([bx, by, carry], yaw, 1.8 * s)
    phase("over_bin")
    robot.move_linear([bx, by, RELEASE_Z + gz], yaw, 0.6 * s)
    phase("release")
    robot.gripper(GRIP_OPEN, 0.5)
    robot.move_linear([bx, by, carry], yaw, 0.6 * s)


def park(robot: Robot, duration: float = 1.8) -> None:
    robot.move_joints(PARK, duration)


assert BIN_WALL_H < RELEASE_Z
