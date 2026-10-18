"""Grasping/sorting success rate over many random layouts (no API calls, no video).

The VLM is replaced by ground-truth boxes projected into the camera image; depth-based
localization, grasp planning and control are exactly the same as in the demo.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from recycle_sort.episode import OracleDetector, run_episode  # noqa: E402
from recycle_sort.layout import demo_layout, random_layout  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 10], help="range [start, stop)")
    ap.add_argument("--demo-mix", action="store_true", help="7 items, two lying on their side")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for seed in range(*args.seeds):
        layout = demo_layout(seed) if args.demo_mix else random_layout(seed)
        log = run_episode(layout, OracleDetector(), verbose=False)
        fails = {n: v for n, v in log["result"].items() if v[0] != v[1]}
        rows.append({"seed": seed, "correct": log["correct"], "total": log["total"],
                     "cycles": log["cycles_used"], "fails": fails,
                     "lying_picks": sum(1 for c in log["cycles"] if c.get("target", {}).get("lying")),
                     "mean_err_mm": log["mean_position_error_mm"]})
        print(f"seed {seed:3d}: {log['correct']}/{log['total']}  cycles {log['cycles_used']:2d}  "
              f"err {log['mean_position_error_mm']:.1f}mm  fails {fails}", flush=True)
    tot = sum(r["total"] for r in rows)
    ok = sum(r["correct"] for r in rows)
    print(f"TOTAL {ok}/{tot} = {ok / tot:.1%}")
    out = args.out or ROOT / "output" / f"batch_eval{'_demo' if args.demo_mix else ''}.json"
    Path(out).write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
