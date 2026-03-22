#!/usr/bin/env python3

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable


def mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def to_100(x_0_5: float | None) -> float | None:
    if x_0_5 is None:
        return None
    return x_0_5 * 20.0


def safe_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class Row:
    model: str
    visual_100: float | None
    audio_100: float | None
    joint_100: float | None
    avg_100: float | None


def extract_model_name(path: str) -> str:
    base = os.path.basename(path)
    if base.startswith("step3_grades_") and base.endswith(".json"):
        return base[len("step3_grades_") : -len(".json")]
    return os.path.splitext(base)[0]


def iter_scores(data: list[dict[str, Any]]) -> tuple[list[float], list[float], list[float], list[float]]:
    visual: list[float] = []
    audio: list[float] = []
    joint: list[float] = []
    all_scores: list[float] = []

    for item in data:
        grades = item.get("grades") or {}
        if not isinstance(grades, dict):
            continue
        for i in range(1, 21):
            key = f"question_{i:02d}"
            v = safe_float(grades.get(key))
            if v is None:
                continue
            all_scores.append(v)
            if 1 <= i <= 5:
                visual.append(v)
            elif 6 <= i <= 10:
                audio.append(v)
            else:
                joint.append(v)

    return visual, audio, joint, all_scores


def fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def print_table(rows: list[Row]) -> None:
    headers = ["model", "visual", "audio", "joint", "avg"]
    body: list[list[str]] = []
    for r in rows:
        body.append([
            r.model,
            fmt(r.visual_100),
            fmt(r.audio_100),
            fmt(r.joint_100),
            fmt(r.avg_100),
        ])

    widths = [len(h) for h in headers]
    for line in body:
        for i, cell in enumerate(line):
            widths[i] = max(widths[i], len(cell))

    def join_line(cols: Iterable[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    print(join_line(headers))
    print(join_line(["-" * w for w in widths]))
    for line in body:
        print(join_line(line))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Summarize QA grading JSONs (step3_grades_*.json) into a CLI table (100-point scale)."
    )
    ap.add_argument(
        "--dir",
        default=os.path.join(os.path.dirname(__file__), "outputs"),
        help="Directory containing grade JSONs (default: evaluation/qa/outputs)",
    )
    ap.add_argument(
        "--pattern",
        default="step3_grades_*.json",
        help="Glob pattern inside --dir (default: step3_grades_*.json)",
    )
    ap.add_argument(
        "--sort",
        choices=["avg", "visual", "audio", "joint", "model"],
        default="avg",
        help="Sort key (default: avg)",
    )
    ap.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse sort (default: false; note avg/visual/audio/joint sort descending by default)",
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        raise SystemExit(f"No files matched: {os.path.join(args.dir, args.pattern)}")

    rows: list[Row] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue

        visual, audio, joint, all_scores = iter_scores(data)
        rows.append(
            Row(
                model=extract_model_name(p),
                visual_100=to_100(mean(visual)),
                audio_100=to_100(mean(audio)),
                joint_100=to_100(mean(joint)),
                avg_100=to_100(mean(all_scores)),
            )
        )

    key = args.sort
    if key == "model":
        rows.sort(key=lambda r: r.model)
        if args.reverse:
            rows.reverse()
    else:
        def getv(r: Row) -> float:
            v = getattr(r, f"{key}_100")
            return -1e18 if v is None else float(v)

        rows.sort(key=getv, reverse=not args.reverse)

    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
