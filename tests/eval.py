"""End-to-end evaluation script for SquatType on the three test clip pairs.

Usage:
    TEST_VIDEO_DIR=/path/to/clips python tests/eval.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from logic import AXIS_CODES, VIEW_AXES, compute_squat_type
from pipeline import (
    normalize_clip,
    recheck_disputed_reps_progressive,
    screen_both_clips,
)


PAIRS: List[Tuple[str, str, str]] = [
    ("Mixed (planted faults)", "front_mixed_720p.mp4", "side_mixed_720p.mp4"),
    ("Clean reference squats", "ref_front_5reps.mp4", "ref_side_5reps.mp4"),
    ("Original 'before' squats", "front_before_720p.mp4", "side_before_720p.mp4"),
]


def _interval_distance(pred_bottom: float, gt_interval: List[float]) -> float:
    start, end = float(gt_interval[0]), float(gt_interval[1])
    if start <= pred_bottom <= end:
        return 0.0
    if pred_bottom < start:
        return start - pred_bottom
    return pred_bottom - end


def match_predicted_rep(
    gt_bottom: List[float],
    pred_reps: List[Dict[str, Any]],
    used_indices: set[int],
    tolerance: float = 0.7,
) -> Optional[Dict[str, Any]]:
    best_idx: Optional[int] = None
    best_dist = float("inf")
    for idx, rep in enumerate(pred_reps):
        if idx in used_indices:
            continue
        dist = _interval_distance(float(rep["bottom"]), gt_bottom)
        if dist <= tolerance and dist < best_dist:
            best_dist = dist
            best_idx = idx
    if best_idx is not None:
        used_indices.add(best_idx)
        return pred_reps[best_idx]
    return None


def score_clip(
    clip_name: str,
    view: str,
    pred_reps: List[Dict[str, Any]],
    gt_entry: Dict[str, Any],
) -> Dict[str, int]:
    axes = VIEW_AXES[view]
    counts = {
        "scored": 0,
        "correct": 0,
        "false_alarm": 0,
        "missed": 0,
        "still_unclear": 0,
    }

    print(f"\n--- {clip_name} ({view}) ---")
    header_axes = " | ".join(f"{ax:<22}" for ax in axes)
    print(f"{'Rep':<5} | {'Bottom':<12} | {header_axes}")
    print("-" * (24 + 25 * len(axes)))

    gt_reps = gt_entry.get("reps")
    if not isinstance(gt_reps, list):
        for rep in pred_reps:
            cells = rep.get("cells", {})
            cell_strs = []
            for ax in axes:
                c = cells.get(ax, {})
                val = c.get("value", "?")
                st = c.get("status", "?")
                if st == "check":
                    counts["still_unclear"] += 1
                cell_strs.append(f"{val} ({st})")
            row_str = " | ".join(f"{s:<22}" for s in cell_strs)
            print(f"{rep['rep']:<5} | {rep['bottom']:<12.1f} | {row_str}")
        print(f"Note: {gt_reps}")
        return counts

    used_indices: set[int] = set()
    for rep_idx, gt_rep in enumerate(gt_reps, start=1):
        gt_bottom = gt_rep["bottom"]
        matched = match_predicted_rep(gt_bottom, pred_reps, used_indices)
        bottom_str = (
            f"{matched['bottom']:.1f}s"
            if matched is not None
            else f"MISS({gt_bottom[0]}-{gt_bottom[1]})"
        )
        cell_strs = []
        for ax in axes:
            exp = gt_rep.get(ax, "either")
            if matched is None:
                if exp != "either":
                    counts["scored"] += 1
                    if exp == AXIS_CODES[ax]["problem"]:
                        counts["missed"] += 1
                    else:
                        counts["false_alarm"] += 1
                cell_strs.append(f"MISS vs {exp}")
                continue

            c = matched.get("cells", {}).get(ax, {})
            pred = c.get("value", "?")
            st = c.get("status", "?")
            if st == "check":
                counts["still_unclear"] += 1

            if exp == "either":
                cell_strs.append(f"{pred}=either ({st[:4]})")
            else:
                counts["scored"] += 1
                if pred == exp:
                    counts["correct"] += 1
                    mark = "OK"
                elif pred == AXIS_CODES[ax]["problem"]:
                    counts["false_alarm"] += 1
                    mark = "FA"
                else:
                    counts["missed"] += 1
                    mark = "MS"
                cell_strs.append(f"{pred}/{exp} {mark} ({st[:4]})")

        row_str = " | ".join(f"{s:<22}" for s in cell_strs)
        print(f"{rep_idx:<5} | {bottom_str:<12} | {row_str}")

    return counts


def main() -> int:
    video_dir_str = os.environ.get("TEST_VIDEO_DIR")
    if video_dir_str:
        video_dir = Path(video_dir_str)
    else:
        video_dir = REPO_ROOT.parent / "test_videos"

    if not video_dir.exists():
        print(f"Error: TEST_VIDEO_DIR not found at {video_dir}", file=sys.stderr)
        return 1

    with open(REPO_ROOT / "answer_key.json", "r", encoding="utf-8") as f:
        answer_key = json.load(f)
    with open(REPO_ROOT / "content.json", "r", encoding="utf-8") as f:
        content = json.load(f)

    pair_summaries = []
    all_passed = True

    for label, front_file, side_file in PAIRS:
        print("\n" + "=" * 78)
        print(f"PAIR: {label} ({front_file} + {side_file})")
        print("=" * 78)
        t0 = time.monotonic()

        front_raw = (video_dir / front_file).read_bytes()
        side_raw = (video_dir / side_file).read_bytes()
        front_norm, _ = normalize_clip(front_raw)
        side_norm, _ = normalize_clip(side_raw)

        screened = screen_both_clips(front_norm, side_norm)
        for _ in recheck_disputed_reps_progressive(front_norm, side_norm, screened):
            pass

        front_reps = screened["front"]
        side_reps = screened["side"]
        pred_type = compute_squat_type(front_reps, side_reps)
        type_name = (content["types"].get(pred_type) or ["Unknown"])[0]
        elapsed = time.monotonic() - t0

        f_counts = score_clip(
            front_file, "front", front_reps, answer_key.get(front_file, {})
        )
        s_counts = score_clip(
            side_file, "side", side_reps, answer_key.get(side_file, {})
        )

        total_scored = f_counts["scored"] + s_counts["scored"]
        total_correct = f_counts["correct"] + s_counts["correct"]
        total_fa = f_counts["false_alarm"] + s_counts["false_alarm"]
        total_missed = f_counts["missed"] + s_counts["missed"]
        total_unclear = f_counts["still_unclear"] + s_counts["still_unclear"]

        expected_type = answer_key.get(front_file, {}).get("expected_type_with_side")
        type_line = f"{pred_type} ({type_name})"
        if expected_type:
            type_line += f" [expected: {expected_type}]"
            if pred_type != expected_type:
                all_passed = False

        if total_scored > 0 and (total_correct != total_scored or total_fa > 0):
            all_passed = False

        print(
            f"\nPair Summary ({elapsed:.1f}s): "
            f"correct={total_correct}/{total_scored}, "
            f"false_alarm={total_fa}, "
            f"missed={total_missed}, "
            f"still_unclear={total_unclear} | "
            f"Predicted Type: {type_line}"
        )

        pair_summaries.append(
            {
                "label": label,
                "scored": total_scored,
                "correct": total_correct,
                "false_alarm": total_fa,
                "missed": total_missed,
                "still_unclear": total_unclear,
                "pred_type": pred_type,
                "type_name": type_name,
                "expected_type": expected_type,
                "elapsed_s": elapsed,
            }
        )

    print("\n" + "=" * 78)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 78)
    for s in pair_summaries:
        score_part = (
            f"{s['correct']}/{s['scored']} correct, "
            f"{s['false_alarm']} false alarms, "
            f"{s['missed']} missed, "
            f"{s['still_unclear']} still unclear"
            if s["scored"] > 0
            else f"not scored per rep ({s['still_unclear']} still unclear)"
        )
        print(
            f"- {s['label']}: {score_part} | "
            f"Type: {s['pred_type']} ({s['type_name']}) [{s['elapsed_s']:.1f}s]"
        )

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
