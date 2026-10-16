from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

VIEW_AXES: Dict[str, Tuple[str, ...]] = {
    "front": ("upper", "knee", "tempo"),
    "side": ("upper", "depth", "tempo"),
}

AXIS_CODES: Dict[str, Dict[str, str]] = {
    "upper": {"good": "S", "problem": "R"},
    "knee": {"good": "A", "problem": "V"},
    "depth": {"good": "D", "problem": "H"},
    "tempo": {"good": "C", "problem": "B"},
}

KOREAN_TO_CODE: Dict[str, Dict[str, Optional[str]]] = {
    "upper": {
        "곧음": "S",
        "말림": "R",
        "판단 불가": None,
    },
    "knee": {
        "정렬": "A",
        "안쪽": "V",
        "판단 불가": None,
    },
    "depth": {
        "깊음": "D",
        "얕음": "H",
        "판단 불가": None,
    },
    "tempo": {
        "충분": "C",
        "짧음": "B",
        "없음": "B",
        "판단 불가": None,
    },
}

RECHECK_AXIS_LINES: Dict[str, str] = {
    "upper": "- upper: 곧음(시선이 정면 또는 2~3m 앞 바닥, 가슴이 펴짐) / 말림(고개를 숙여 발 바로 앞을 보거나 등 위쪽이 둥글게 말림)",
    "knee": "- knee: 정렬(무릎이 두 번째 발가락 방향이거나 그보다 바깥) / 안쪽(두 무릎 사이 간격이 좁아지거나 무릎이 엄지발가락 쪽이나 그 안쪽으로 모임)",
    "depth": "- depth: 깊음(고관절이 무릎 윗부분과 같은 높이이거나 더 낮음) / 얕음(고관절이 무릎보다 확실히 높음)",
    "tempo": "- pause: 충분(가장 낮은 지점에서 1초 이상 멈춤) / 짧음(멈췄지만 1초 미만) / 없음(바로 올라옴)",
}


def build_recheck_prompt(view: str, disputed_axes: Sequence[str]) -> str:
    """Build the verbatim Korean RECHECK prompt for the disputed axes of one rep."""
    view_kr = "정면" if view == "front" else "측면"
    axis_lines = "\n".join(RECHECK_AXIS_LINES[ax] for ax in disputed_axes)
    json_keys = [
        ("pause" if ax == "tempo" else ax)
        for ax in disputed_axes
    ]
    json_example = "{" + ", ".join(f'"{k}": "..."' for k in json_keys) + "}"
    return (
        f"첨부한 영상 구간에는 {view_kr}에서 찍은 맨몸 스쿼트 반복이 한 번 있어. 가장 낮은 지점을 기준으로 판정해.\n"
        f"{axis_lines}\n"
        f"바르게 한 반복은 바르다고 판정해. 문제를 억지로 찾지 마.\n"
        f"JSON만 출력해: {json_example}"
    )


def normalize_korean_value(axis: str, raw_value: Optional[str]) -> Optional[str]:
    """Convert a Korean model judgment string into a single-letter code.

    Reads only the text before any '(' and strips spaces.
    Returns None for '판단 불가' or unrecognized values.
    """
    if not raw_value or not isinstance(raw_value, str):
        return None
    canonical_axis = "tempo" if axis == "pause" else axis
    head = raw_value.split("(", 1)[0].strip()
    axis_map = KOREAN_TO_CODE.get(canonical_axis, {})
    if head in axis_map:
        return axis_map[head]
    valid_codes = AXIS_CODES.get(canonical_axis, {})
    if head in (valid_codes.get("good"), valid_codes.get("problem")):
        return head
    return None


def normalize_rep_dict(raw_rep: Dict[str, Any], view: str) -> Dict[str, Any]:
    """Normalize a single rep dictionary from a model run."""
    bottom = float(raw_rep.get("bottom", 0.0))
    pause_or_tempo = raw_rep.get("pause", raw_rep.get("tempo"))
    out: Dict[str, Any] = {
        "rep": int(raw_rep.get("rep", 0)),
        "bottom": bottom,
    }
    for axis in VIEW_AXES[view]:
        raw_val = pause_or_tempo if axis == "tempo" else raw_rep.get(axis)
        out[axis] = normalize_korean_value(axis, raw_val)
    return out


def align_reps_across_runs(
    runs: Sequence[Sequence[Dict[str, Any]]],
    view: str,
    max_offset_seconds: float = 0.7,
) -> List[Dict[str, Any]]:
    """Align reps across successful Flash runs for a single view.

    - Drops any empty/failed runs.
    - Takes the most common rep count across successful runs.
    - Uses the first run with that count as the anchor.
    - Matches each successful run's reps to the anchor by nearest bottom within 0.7 s.
    """
    normalized_runs: List[List[Dict[str, Any]]] = [
        [normalize_rep_dict(r, view) for r in run]
        for run in runs
        if run
    ]
    if not normalized_runs:
        return []

    counts = Counter(len(run) for run in normalized_runs)
    mode_count, _ = counts.most_common(1)[0]
    anchor_run = next(run for run in normalized_runs if len(run) == mode_count)

    aligned: List[Dict[str, Any]] = []
    num_successful = len(normalized_runs)

    for idx, anchor_rep in enumerate(anchor_run, start=1):
        anchor_bottom = float(anchor_rep["bottom"])
        votes_by_axis: Dict[str, List[Optional[str]]] = {
            axis: [] for axis in VIEW_AXES[view]
        }

        for norm_run in normalized_runs:
            nearest = min(
                norm_run,
                key=lambda r: abs(float(r["bottom"]) - anchor_bottom),
            )
            if abs(float(nearest["bottom"]) - anchor_bottom) <= max_offset_seconds:
                for axis in VIEW_AXES[view]:
                    votes_by_axis[axis].append(nearest.get(axis))
            else:
                for axis in VIEW_AXES[view]:
                    votes_by_axis[axis].append(None)

        aligned.append(
            {
                "rep": idx,
                "bottom": anchor_bottom,
                "successful_runs": num_successful,
                "votes": votes_by_axis,
            }
        )

    return aligned


def majority_vote(votes: Sequence[Optional[str]], axis: str) -> str:
    """Compute majority value for an axis; ties break toward the problem value."""
    good_code = AXIS_CODES[axis]["good"]
    problem_code = AXIS_CODES[axis]["problem"]
    valid = [v for v in votes if v in (good_code, problem_code)]
    good_count = sum(1 for v in valid if v == good_code)
    problem_count = sum(1 for v in valid if v == problem_code)
    if problem_count >= good_count:
        return problem_code
    return good_code


def vote_aligned_reps(
    aligned_reps: Sequence[Dict[str, Any]],
    view: str,
    expected_runs: Optional[int] = None,
    min_successful_runs: int = 3,
) -> List[Dict[str, Any]]:
    """Evaluate votes per (rep, axis) cell among successful runs.

    Unanimous among successful runs (with no missing aligned reps) -> Confirmed,
    provided at least `min_successful_runs` (3) runs succeeded.
    If fewer than 3 runs succeeded, or votes split / rep missing -> disputed.
    """
    axes = VIEW_AXES[view]
    results: List[Dict[str, Any]] = []

    for item in aligned_reps:
        cells: Dict[str, Dict[str, Any]] = {}
        disputed_axes: List[str] = []
        first_axis_votes = next(iter(item["votes"].values()), [])
        target_runs = (
            expected_runs
            if expected_runs is not None
            else int(item.get("successful_runs", len(first_axis_votes)))
        )

        for axis in axes:
            votes: List[Optional[str]] = list(item["votes"].get(axis, []))
            good_code = AXIS_CODES[axis]["good"]
            problem_code = AXIS_CODES[axis]["problem"]
            valid_votes = [v for v in votes if v in (good_code, problem_code)]

            unanimous = (
                target_runs >= min_successful_runs
                and len(valid_votes) == target_runs
                and len(set(valid_votes)) == 1
            )
            maj = majority_vote(votes, axis)

            if unanimous:
                cells[axis] = {
                    "value": valid_votes[0],
                    "majority_value": maj,
                    "status": "confirmed",
                    "disputed": False,
                    "votes": votes,
                    "user_checked": None,
                    "timeout_reason": None,
                }
            else:
                disputed_axes.append(axis)
                cells[axis] = {
                    "value": maj,
                    "majority_value": maj,
                    "status": "verifying",
                    "disputed": True,
                    "votes": votes,
                    "user_checked": None,
                    "timeout_reason": None,
                }

        results.append(
            {
                "rep": item["rep"],
                "bottom": item["bottom"],
                "cells": cells,
                "disputed_axes": disputed_axes,
            }
        )

    return results


def apply_argon_result(
    rep_entry: Dict[str, Any],
    raw_argon_json: Optional[Dict[str, Any]],
) -> None:
    """Update a disputed rep's cells in place from Argon's parsed JSON response.

    If raw_argon_json is None or an axis is missing/unparseable, falls back
    to majority_value with status 'check' and sets timeout_reason.
    """
    rep_num = rep_entry["rep"]
    disputed_axes: List[str] = rep_entry.get("disputed_axes", [])
    cells = rep_entry["cells"]

    for axis in disputed_axes:
        cell = cells[axis]
        code: Optional[str] = None
        if isinstance(raw_argon_json, dict):
            raw_val = (
                raw_argon_json.get("pause", raw_argon_json.get("tempo"))
                if axis == "tempo"
                else raw_argon_json.get(axis)
            )
            code = normalize_korean_value(axis, raw_val)

        if code is not None:
            cell["value"] = code
            cell["status"] = "verified"
            cell["timeout_reason"] = None
        else:
            cell["value"] = cell["majority_value"]
            cell["status"] = "check"
            cell["timeout_reason"] = f"Argon took too long on rep {rep_num}."


def compute_view_axis_letter(reps: Sequence[Dict[str, Any]], axis: str) -> str:
    """Return problem letter if more than half of the view's reps have problem."""
    good_code = AXIS_CODES[axis]["good"]
    problem_code = AXIS_CODES[axis]["problem"]
    if not reps:
        return good_code
    problem_count = sum(
        1
        for r in reps
        if r.get("cells", {}).get(axis, {}).get("value") == problem_code
    )
    if problem_count > len(reps) / 2.0:
        return problem_code
    return good_code


def compute_squat_type(
    front_reps: Sequence[Dict[str, Any]],
    side_reps: Sequence[Dict[str, Any]],
) -> str:
    """Compute the 4-letter squat type (upper-knee-depth-tempo)."""
    upper_front = compute_view_axis_letter(front_reps, "upper")
    upper_side = compute_view_axis_letter(side_reps, "upper")
    upper = "R" if ("R" in (upper_front, upper_side)) else "S"

    knee = compute_view_axis_letter(front_reps, "knee")
    depth = compute_view_axis_letter(side_reps, "depth")

    tempo_front = compute_view_axis_letter(front_reps, "tempo")
    tempo_side = compute_view_axis_letter(side_reps, "tempo")
    tempo = "B" if ("B" in (tempo_front, tempo_side)) else "C"

    return f"{upper}{knee}{depth}{tempo}"


def format_rep_list(rep_numbers: Sequence[int]) -> str:
    """Format rep numbers into natural English, e.g. 'rep 2' or 'reps 2 and 4'."""
    nums = sorted(set(int(x) for x in rep_numbers))
    if not nums:
        return ""
    if len(nums) == 1:
        return f"rep {nums[0]}"
    if len(nums) == 2:
        return f"reps {nums[0]} and {nums[1]}"
    leading = ", ".join(str(n) for n in nums[:-1])
    return f"reps {leading} and {nums[-1]}"


def build_why_line(
    problem_letter: str,
    front_reps: Sequence[Dict[str, Any]],
    side_reps: Sequence[Dict[str, Any]],
) -> str:
    """Build a factual 'Why: ...' explanation with specific rep numbers."""
    letter_to_axis = {"R": "upper", "V": "knee", "H": "depth", "B": "tempo"}
    axis = letter_to_axis[problem_letter]

    front_flagged = [
        r["rep"]
        for r in front_reps
        if r.get("cells", {}).get(axis, {}).get("value") == problem_letter
    ]
    side_flagged = [
        r["rep"]
        for r in side_reps
        if r.get("cells", {}).get(axis, {}).get("value") == problem_letter
    ]

    if problem_letter == "V":
        rep_str = format_rep_list(front_flagged)
        return f"Why: knees caved in on {rep_str}." if rep_str else "Why: knees caved in."
    if problem_letter == "H":
        rep_str = format_rep_list(side_flagged)
        return f"Why: depth was half on {rep_str}." if rep_str else "Why: depth was half."
    if problem_letter == "R":
        combined = sorted(set(front_flagged) | set(side_flagged))
        rep_str = format_rep_list(combined)
        return f"Why: upper body rounded on {rep_str}." if rep_str else "Why: upper body rounded."
    if problem_letter == "B":
        combined = sorted(set(front_flagged) | set(side_flagged))
        rep_str = format_rep_list(combined)
        return f"Why: tempo was bouncy on {rep_str}." if rep_str else "Why: tempo was bouncy."
    return ""


def select_recommendations(
    squat_type: str,
    pain: str,
    experience: str,
    content: Dict[str, Any],
    front_reps: Sequence[Dict[str, Any]] = (),
    side_reps: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """Select exercise recommendations based on squat type and survey answers."""
    if pain == "Yes":
        return {
            "pain_message": content["messages"]["pain"],
            "level": None,
            "items": [],
        }

    recs_cfg = content["recommendations"]
    priority: List[str] = recs_cfg["priority"]
    problem_letters = [p for p in priority if p in squat_type]

    level = experience if experience in ("Beginner", "Sometimes", "Regular") else "Beginner"
    if level == "Regular" and len(problem_letters) >= 3:
        level = "Sometimes"

    if not problem_letters:
        ex = recs_cfg["all_clear"][level]
        return {
            "pain_message": None,
            "level": level,
            "items": [
                {
                    "letter": None,
                    "name": ex["name"],
                    "line": ex["line"],
                    "why": None,
                }
            ],
        }

    selected_items: List[Dict[str, Any]] = []
    for letter in problem_letters[:2]:
        ex = recs_cfg[letter][level]
        selected_items.append(
            {
                "letter": letter,
                "name": ex["name"],
                "line": ex["line"],
                "why": build_why_line(letter, front_reps, side_reps),
            }
        )

    return {
        "pain_message": None,
        "level": level,
        "items": selected_items,
    }
