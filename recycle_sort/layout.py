"""Random table layouts of recyclable items."""

from __future__ import annotations

import numpy as np

from .scene import ITEM_TYPES, Item

ACCENTS = {
    "pet_bottle": [(0.10, 0.55, 0.25), (0.95, 0.50, 0.10), (0.20, 0.50, 0.90)],
    "can": [(0.80, 0.10, 0.12), (0.10, 0.30, 0.75), (0.15, 0.15, 0.15), (0.20, 0.60, 0.30)],
    "glass_bottle": [(0.93, 0.89, 0.76)],
    "paper_carton": [(0.15, 0.35, 0.80), (0.90, 0.40, 0.60), (0.45, 0.30, 0.20)],
}
DEFAULT_MIX = ["pet_bottle", "pet_bottle", "can", "can", "glass_bottle", "paper_carton"]
# Demo mix: one bottle and one can lying on their side, plus an extra carton.
DEMO_MIX = [("pet_bottle", False), ("pet_bottle", True), ("can", False), ("can", True),
            ("glass_bottle", False), ("paper_carton", False), ("paper_carton", False)]


def random_layout(seed: int, kinds=DEFAULT_MIX, x_range=(0.36, 0.64), y_range=(-0.20, 0.20),
                  min_dist: float = 0.12, restarts: int = 500) -> list[Item]:
    """Place items at random with a minimum clearance between their footprints.

    kinds: list of kind names, or of (kind, lying) pairs. When an item cannot be
    placed, the whole layout is drawn again.
    """
    rng = np.random.default_rng(seed)
    specs = [(k, False) if isinstance(k, str) else tuple(k) for k in kinds]
    assert all(k in ITEM_TYPES for k, _ in specs)
    # Place lying (long) items first: they are the hardest to fit.
    specs.sort(key=lambda s: not s[1])
    for _ in range(restarts):
        placed = _try_layout(rng, specs, x_range, y_range, min_dist)
        if placed:
            order = rng.permutation(len(placed))   # shuffle names so ids don't reveal kinds
            return [Item(f"item{i}", p.kind, p.x, p.y, p.yaw, p.accent, p.lying)
                    for i, p in zip(order, placed)]
    raise RuntimeError("could not place all items; relax min_dist or ranges")


def _try_layout(rng, specs, x_range, y_range, min_dist, tries_per_item: int = 300):
    placed: list[Item] = []
    for kind, lying in specs:
        acc = ACCENTS[kind][rng.integers(len(ACCENTS[kind]))]
        for _ in range(tries_per_item):
            cand = Item("", kind, float(rng.uniform(*x_range)), float(rng.uniform(*y_range)),
                        yaw=float(rng.uniform(-np.pi, np.pi)), accent=acc, lying=lying)
            fp = cand.footprint()
            inside = (x_range[0] - 0.02 <= fp[:, 0].min() and fp[:, 0].max() <= x_range[1] + 0.02
                      and y_range[0] - 0.03 <= fp[:, 1].min() and fp[:, 1].max() <= y_range[1] + 0.03)
            if inside and all(np.min(np.linalg.norm(fp[:, None] - p.footprint()[None], axis=2))
                              >= min_dist for p in placed):
                placed.append(cand)
                break
        else:
            return None
    return placed


def demo_layout(seed: int) -> list[Item]:
    """Seven items, two of them lying on their side, on a slightly wider area."""
    return random_layout(seed, DEMO_MIX, x_range=(0.34, 0.66), y_range=(-0.21, 0.21))
