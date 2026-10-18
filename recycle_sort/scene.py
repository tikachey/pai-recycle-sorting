"""Build the sorting scene (Panda arm, table, bins, recyclables) with MjSpec."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .assets import lathe_mesh

ROOT = Path(__file__).resolve().parent.parent
PANDA_XML = ROOT / "third_party" / "franka_emika_panda" / "panda.xml"

# Recycling categories (Japanese municipal style: PET / cans / glass / paper cartons)
CATEGORIES = {
    "pet": dict(label="PET", rgb=(0.18, 0.42, 0.85), bin_xy=(0.22, 0.42)),
    "can": dict(label="CAN", rgb=(0.95, 0.70, 0.10), bin_xy=(0.50, 0.42)),
    "glass": dict(label="GLASS", rgb=(0.20, 0.60, 0.28), bin_xy=(0.22, -0.42)),
    "paper": dict(label="PAPER", rgb=(0.60, 0.40, 0.22), bin_xy=(0.50, -0.42)),
}
BIN_HALF = 0.11       # inner half width of a bin [m] (22 cm, fits a lying 2 L-class bottle)
BIN_WALL_H = 0.10     # wall height [m]
# Drop positions inside a bin, used in turn so that items do not pile up.
BIN_SLOTS = [(0.045, 0.045), (-0.045, -0.045), (0.045, -0.045), (-0.045, 0.045), (0.0, 0.0)]


@dataclass(frozen=True)
class ItemType:
    category: str
    height: float     # total height [m]
    grasp_z: float    # TCP height above the item base when grasping [m]
    mass: float
    radius: float     # footprint radius used for spacing [m]
    narrow_axis: bool = False  # True: must close the gripper along local y


ITEM_TYPES = {
    "pet_bottle": ItemType("pet", 0.198, 0.184, 0.03, 0.034),
    "can": ItemType("can", 0.122, 0.095, 0.02, 0.034),
    "glass_bottle": ItemType("glass", 0.198, 0.176, 0.19, 0.032),
    "paper_carton": ItemType("paper", 0.100, 0.072, 0.015, 0.035, narrow_axis=True),
}


@dataclass
class Item:
    name: str
    kind: str
    x: float
    y: float
    yaw: float = 0.0
    accent: tuple[float, float, float] = (0.8, 0.1, 0.1)
    lying: bool = False   # placed on its side; (x, y) is then the centre and yaw the axis

    @property
    def type(self) -> ItemType:
        return ITEM_TYPES[self.kind]

    def footprint(self, n: int = 5) -> np.ndarray:
        """Points along the item's footprint (one point if standing, its axis if lying)."""
        if not self.lying:
            return np.array([[self.x, self.y]])
        d = np.array([np.cos(self.yaw), np.sin(self.yaw)])
        s = np.linspace(-0.5, 0.5, n)[:, None] * self.type.height
        return np.array([self.x, self.y]) + s * d


@dataclass
class Scene:
    spec: mujoco.MjSpec
    model: mujoco.MjModel
    items: list[Item] = field(default_factory=list)


def _quat_z(yaw: float) -> list[float]:
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def _quat_mul(a, b) -> list[float]:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out.tolist()


def _visual(body, **kw):
    g = body.add_geom(contype=0, conaffinity=0, group=0, **kw)
    return g


def _collision(body, **kw):
    g = body.add_geom(group=3, condim=4, friction=[1.2, 0.02, 0.001],
                      rgba=[0.5, 0.5, 0.5, 1], **kw)
    return g


def _collision_cyl(spec, body, radius: float, half: float, zc: float):
    """Cylinder collision as a convex prism mesh.

    MuJoCo's native cylinder contacts let the small box-shaped finger pads sink deep
    into light items; a convex mesh hull gives clean, shallow contacts.
    """
    name = f"{body.name}_hull{len(body.geoms)}"
    _add_mesh(spec, name, [(radius, zc - half), (radius, zc + half)], segments=24)
    return _collision(body, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name)


def _add_mesh(spec, name, profile, segments: int = 32):
    v, f = lathe_mesh(profile, segments)
    spec.add_mesh(name=name, uservert=v.ravel().tolist(), userface=f.ravel().tolist())


def _add_environment(spec: mujoco.MjSpec) -> None:
    wb = spec.worldbody
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 8
    spec.visual.headlight.ambient = [0.22, 0.22, 0.22]
    spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]

    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.85, 0.90, 0.96], rgb2=[0.45, 0.52, 0.62],
                     width=512, height=3072)
    spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.30, 0.32, 0.36], rgb2=[0.26, 0.28, 0.31],
                     width=300, height=300)
    floor_mat = spec.add_material(name="floor_mat", texrepeat=[6, 6], reflectance=0.1)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"

    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                pos=[0, 0, -0.75], material="floor_mat")
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.55, 0.80, 0.025],
                pos=[0.35, 0, -0.025], rgba=[0.80, 0.70, 0.56, 1], friction=[1.0, 0.02, 0.001])
    for sx in (-1, 1):
        for sy in (-1, 1):
            wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.36],
                        pos=[0.35 + sx * 0.5, sy * 0.75, -0.41], rgba=[0.45, 0.38, 0.30, 1],
                        contype=0, conaffinity=0)

    wb.add_light(name="key", pos=[0.6, -0.3, 2.0], dir=[-0.15, 0.1, -1],
                 diffuse=[0.42, 0.42, 0.42], specular=[0.15, 0.15, 0.15], castshadow=True,
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    wb.add_light(name="fill", pos=[1.5, 1.0, 1.5], dir=[-1, -0.6, -0.8],
                 diffuse=[0.15, 0.15, 0.15], castshadow=False,
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)

    # Camera targets and cameras
    look = wb.add_body(name="video_target", pos=[0.38, 0.0, 0.22])
    wb.add_camera(name="video", pos=[1.65, -1.10, 1.20], fovy=44,
                  mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody="video_target")
    wb.add_camera(name="video_wide", pos=[1.25, 0.0, 1.25], fovy=55,
                  mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody="video_target")
    del look
    # Fixed RGB-D camera for the AI, looking at the work area from the far side of the table.
    wb.add_body(name="ai_target", pos=[0.45, 0.0, 0.03])
    wb.add_camera(name="ai_cam", pos=[1.12, 0.0, 0.78], fovy=58,
                  mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody="ai_target")
    # Top-down camera for the AI: image up = +x (away from the robot), image right = -y.
    wb.add_camera(name="overhead", pos=[0.40, 0.0, 1.05], fovy=50,
                  quat=[np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)])


def _add_bins(spec: mujoco.MjSpec) -> None:
    wb = spec.worldbody
    t = 0.006
    for key, c in CATEGORIES.items():
        x, y = c["bin_xy"]
        rgba = [*c["rgb"], 1]
        b = wb.add_body(name=f"bin_{key}", pos=[x, y, 0])
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[BIN_HALF + t, BIN_HALF + t, t / 2],
                   pos=[0, 0, t / 2], rgba=[*(0.75 * np.array(c["rgb"])), 1])
        h = BIN_WALL_H / 2
        for sx, sy, size in (
            (1, 0, [t / 2, BIN_HALF + t, h]), (-1, 0, [t / 2, BIN_HALF + t, h]),
            (0, 1, [BIN_HALF + t, t / 2, h]), (0, -1, [BIN_HALF + t, t / 2, h]),
        ):
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size,
                       pos=[sx * (BIN_HALF + t / 2), sy * (BIN_HALF + t / 2), h], rgba=rgba)


def _add_item(spec: mujoco.MjSpec, item: Item) -> None:
    t = item.type
    if item.lying:
        # Lay the item on its side: local z (base -> top) along the yaw direction.
        d = np.array([np.cos(item.yaw), np.sin(item.yaw)])
        base = np.array([item.x, item.y]) - 0.5 * t.height * d
        quat = _quat_mul(_quat_z(item.yaw), [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])
        body = spec.worldbody.add_body(name=item.name, pos=[*base, t.radius + 0.002], quat=quat)
    else:
        body = spec.worldbody.add_body(name=item.name, pos=[item.x, item.y, 0.002],
                                       quat=_quat_z(item.yaw))
    body.add_freejoint(name=f"{item.name}_joint")
    body.explicitinertial = True
    body.mass = t.mass
    r, h = t.radius, t.height
    i_xy = t.mass * (3 * r * r + h * h) / 12
    body.inertia = [i_xy, i_xy, t.mass * r * r / 2]
    body.ipos = [0, 0, h / 2]
    acc = [*item.accent, 1]
    cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
    mesh = mujoco.mjtGeom.mjGEOM_MESH

    if item.kind == "pet_bottle":
        _add_mesh(spec, f"{item.name}_mesh", [
            (0.030, 0.0), (0.033, 0.005), (0.033, 0.06), (0.031, 0.07), (0.033, 0.08),
            (0.033, 0.14), (0.028, 0.155), (0.018, 0.166), (0.0125, 0.172), (0.0125, 0.18)])
        _visual(body, type=mesh, meshname=f"{item.name}_mesh", rgba=[0.78, 0.90, 0.98, 0.45])
        _visual(body, type=cyl, size=[0.0336, 0.0225], pos=[0, 0, 0.1075], rgba=acc)
        _visual(body, type=cyl, size=[0.0148, 0.010], pos=[0, 0, 0.188], rgba=[0.95, 0.95, 0.95, 1])
        _collision_cyl(spec, body, 0.033, 0.0725, 0.0725)
        _collision_cyl(spec, body, 0.022, 0.0125, 0.1575)
        _collision_cyl(spec, body, 0.0148, 0.014, 0.184)
    elif item.kind == "can":
        _add_mesh(spec, f"{item.name}_mesh", [
            (0.026, 0.0), (0.033, 0.008), (0.033, 0.108), (0.029, 0.118), (0.027, 0.122)])
        spec.add_material(name=f"{item.name}_metal", rgba=[0.82, 0.83, 0.86, 1],
                          specular=0.9, shininess=0.8, reflectance=0.2)
        _visual(body, type=mesh, meshname=f"{item.name}_mesh", material=f"{item.name}_metal")
        _visual(body, type=cyl, size=[0.0334, 0.0465], pos=[0, 0, 0.0585], rgba=acc)
        _collision_cyl(spec, body, 0.033, 0.061, 0.061)
    elif item.kind == "glass_bottle":
        _add_mesh(spec, f"{item.name}_mesh", [
            (0.028, 0.0), (0.031, 0.004), (0.031, 0.11), (0.027, 0.125), (0.016, 0.145),
            (0.0125, 0.155), (0.0125, 0.185), (0.014, 0.188), (0.014, 0.192)])
        _visual(body, type=mesh, meshname=f"{item.name}_mesh", rgba=[0.40, 0.21, 0.06, 0.88])
        _visual(body, type=cyl, size=[0.0316, 0.025], pos=[0, 0, 0.065], rgba=[0.93, 0.89, 0.76, 1])
        _visual(body, type=cyl, size=[0.0148, 0.004], pos=[0, 0, 0.194], rgba=[0.80, 0.65, 0.20, 1])
        _collision_cyl(spec, body, 0.031, 0.0575, 0.0575)
        _collision_cyl(spec, body, 0.020, 0.020, 0.135)
        _collision_cyl(spec, body, 0.0128, 0.0215, 0.1765)
    elif item.kind == "paper_carton":
        box = mujoco.mjtGeom.mjGEOM_BOX
        _visual(body, type=box, size=[0.0275, 0.020, 0.050], pos=[0, 0, 0.050],
                rgba=[0.96, 0.96, 0.94, 1])
        for sy in (-1, 1):
            _visual(body, type=box, size=[0.022, 0.0005, 0.030],
                    pos=[0, sy * 0.0205, 0.045], rgba=acc)
        _visual(body, type=cyl, size=[0.004, 0.006], pos=[0.012, 0.0, 0.104],
                rgba=[0.95, 0.95, 0.95, 1])
        _collision(body, type=box, size=[0.0275, 0.020, 0.050], pos=[0, 0, 0.050])
    else:
        raise ValueError(item.kind)


def build_scene(items: list[Item]) -> Scene:
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 3
    for b in spec.bodies:
        if b.name.startswith("link") or b.name in ("hand", "left_finger", "right_finger"):
            b.gravcomp = 1
    # The stock gripper squeezes with only ~1 N, too weak for a 190 g glass bottle.
    # Raise the servo stiffness (keeping ctrl 255 = 4 cm per finger) for ~20 N.
    grip = spec.actuator("actuator8")
    kp = 800.0
    grip.gainprm = [0.04 * kp / 255] + [0.0] * 9
    grip.biasprm = [0.0, -kp, -40.0] + [0.0] * 7
    spec.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.005, 0, 0],
                               rgba=[1, 0, 0, 0], group=4)
    _add_environment(spec)
    _add_bins(spec)
    for item in items:
        _add_item(spec, item)
    model = spec.compile()
    return Scene(spec=spec, model=model, items=list(items))


def which_bin(data: mujoco.MjData, model: mujoco.MjModel, name: str) -> str | None:
    """The bin that holds the item's centre of mass, if any.

    A bottle standing inside a bin has its centre of mass slightly above the wall top,
    while an item still being carried is far higher, hence the 5 cm allowance.
    """
    pos = data.xipos[model.body(name).id]
    for key, c in CATEGORIES.items():
        bx, by = c["bin_xy"]
        if abs(pos[0] - bx) < BIN_HALF and abs(pos[1] - by) < BIN_HALF \
                and pos[2] < BIN_WALL_H + 0.05:
            return key
    return None
