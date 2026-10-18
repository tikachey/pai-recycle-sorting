"""AI recycling sorter demo: observe -> detect (Gemini Robotics-ER) -> pick & sort -> repeat.

Usage:
    python run_demo.py --seed 3 --env-file .env
    python run_demo.py --seed 3 --offline      # replay cached Gemini responses
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from recycle_sort.episode import GeminiDetector, run_episode
from recycle_sort.layout import demo_layout, random_layout

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--offline", action="store_true", help="use cached Gemini responses only")
    ap.add_argument("--max-cycles", type=int, default=12)
    ap.add_argument("--simple", action="store_true", help="6 standing items instead of the demo mix")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    load_dotenv(args.env_file)

    out = Path(args.out) if args.out else ROOT / "output" / f"run_seed{args.seed}"
    detector = GeminiDetector(ROOT / "output" / "gemini_cache", offline=args.offline)
    layout = random_layout(args.seed) if args.simple else demo_layout(args.seed)
    log = run_episode(layout, detector, out_dir=out, record=True,
                      max_cycles=args.max_cycles)
    log["seed"] = args.seed
    (out / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1, default=str),
                                  encoding="utf-8")
    print(f"correct {log['correct']}/{log['total']}, AI calls {log['detector_calls']}, "
          f"video {log['video_s']:.1f}s -> {out / 'demo.mp4'}")


if __name__ == "__main__":
    main()
