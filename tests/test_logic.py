import json
from pathlib import Path
import time
from unittest.mock import patch

from logic import (
    align_reps_across_runs,
    apply_argon_result,
    apply_group_scene_check,
    apply_user_scene_check,
    build_recheck_prompt,
    build_scene_check_items,
    compute_scene_seek_seconds,
    compute_squat_type,
    count_pending_checks,
    format_provisional_line,
    format_rep_span,
    is_cell_flagged,
    majority_vote,
    normalize_korean_value,
    select_recommendations,
    vote_aligned_reps,
)
from pipeline import (
    recheck_disputed_reps_progressive,
    screen_both_clips,
    strip_json_fences,
)

CONTENT_PATH = Path(__file__).parent.parent / "content.json"
with open(CONTENT_PATH, "r", encoding="utf-8") as f:
    CONTENT = json.load(f)


def test_strip_json_fences_with_leading_sentence():
    raw = 'Here is the JSON analysis for your clip:\n{"reps": [{"rep": 1, "bottom": 2.5}]}'
    cleaned = strip_json_fences(raw)
    parsed = json.loads(cleaned)
    assert parsed == {"reps": [{"rep": 1, "bottom": 2.5}]}

    raw_array = "Sure! Below is the list:\n[{\"rep\": 1}]\nHope this helps."
    assert json.loads(strip_json_fences(raw_array)) == [{"rep": 1}]


def test_normalize_korean_value_strips_parentheses_and_spaces():
    assert normalize_korean_value("upper", " 곧음 ") == "S"
    assert normalize_korean_value("upper", "말림(고개를 숙여 발 바로 앞을 보거나)") == "R"
    assert normalize_korean_value("knee", "정렬 (무릎이 두 번째 발가락 방향)") == "A"
    assert normalize_korean_value("knee", "안쪽") == "V"
    assert normalize_korean_value("depth", "깊음(고관절이 무릎 윗부분과 같은 높이)") == "D"
    assert normalize_korean_value("depth", "얕음") == "H"
    assert normalize_korean_value("tempo", "충분") == "C"
    assert normalize_korean_value("pause", "짧음(멈췄지만 1초 미만)") == "B"
    assert normalize_korean_value("pause", "없음(바로 올라옴)") == "B"
    assert normalize_korean_value("depth", "판단 불가") is None
    assert normalize_korean_value("knee", "판단 불가") is None


def test_build_recheck_prompt():
    prompt = build_recheck_prompt("front", ["knee", "tempo"])
    assert "정면에서 찍은 맨몸 스쿼트 반복이 한 번 있어" in prompt
    assert "- knee: 정렬" in prompt
    assert "- pause: 충분" in prompt
    assert 'JSON만 출력해: {"knee": "...", "pause": "..."}' in prompt


def test_apply_argon_result_success_and_timeout():
    aligned = [
        {
            "rep": 2,
            "bottom": 8.8,
            "successful_runs": 5,
            "votes": {
                "upper": ["S", "S", "S", "S", "S"],
                "knee": ["V", "V", "V", "A", "A"],
                "tempo": ["B", "B", "B", "B", "B"],
            },
        }
    ]
    voted = vote_aligned_reps(aligned, view="front")
    apply_argon_result(voted[0], {"knee": "안쪽(두 무릎 사이 간격이 좁아지거나)"}, view="front")
    assert voted[0]["cells"]["knee"]["value"] == "V"
    assert voted[0]["cells"]["knee"]["status"] == "verified"
    assert voted[0]["cells"]["knee"]["timeout_reason"] is None

    voted_timeout = vote_aligned_reps(aligned, view="front")
    apply_argon_result(voted_timeout[0], None, view="front")
    assert voted_timeout[0]["cells"]["knee"]["value"] == "V"
    assert voted_timeout[0]["cells"]["knee"]["status"] == "check"
    assert voted_timeout[0]["cells"]["knee"]["timeout_reason"] == "Argon took too long on front rep 2."


def test_recheck_timeout_does_not_wait_for_slow_thread():
    aligned = [
        {
            "rep": 3,
            "bottom": 12.0,
            "successful_runs": 5,
            "votes": {
                "upper": ["S", "S", "S", "S", "S"],
                "depth": ["D", "H", "H", "D", "H"],
                "tempo": ["C", "C", "C", "C", "C"],
            },
        }
    ]
    screened = {
        "front": [],
        "side": vote_aligned_reps(aligned, view="side"),
    }

    def slow_argon(*args, **kwargs):
        time.sleep(0.6)
        return {"depth": "깊음"}

    t0 = time.monotonic()
    with patch("pipeline.call_argon_recheck_once", side_effect=slow_argon):
        events = list(
            recheck_disputed_reps_progressive(
                b"front", b"side", screened, timeout_seconds=0.15
            )
        )
    elapsed = time.monotonic() - t0
    assert elapsed < 0.45
    assert len(events) == 1
    cell = screened["side"][0]["cells"]["depth"]
    assert cell["status"] == "check"
    assert cell["value"] == "H"
    assert cell["timeout_reason"] == "Argon took too long on side rep 3."


def test_align_reps_across_runs_mode_and_threshold():
    run1 = [
        {"rep": 1, "bottom": 2.5, "upper": "곧음", "knee": "정렬", "pause": "충분"},
        {"rep": 2, "bottom": 8.9, "upper": "곧음", "knee": "안쪽", "pause": "짧음"},
    ]
    run2 = [
        {"rep": 1, "bottom": 2.6, "upper": "곧음", "knee": "정렬", "pause": "충분"},
        {"rep": 2, "bottom": 9.1, "upper": "곧음", "knee": "안쪽", "pause": "짧음"},
    ]
    run3 = [
        {"rep": 1, "bottom": 2.4, "upper": "곧음", "knee": "정렬", "pause": "충분"},
        {"rep": 2, "bottom": 5.5, "upper": "곧음", "knee": "정렬", "pause": "충분"},
        {"rep": 3, "bottom": 10.0, "upper": "곧음", "knee": "안쪽", "pause": "짧음"},
    ]

    aligned = align_reps_across_runs([run3, run1, run2], view="front")
    assert len(aligned) == 2
    assert aligned[0]["bottom"] == 2.5
    assert aligned[0]["votes"]["upper"] == ["S", "S", "S"]
    assert aligned[1]["bottom"] == 8.9
    assert aligned[1]["votes"]["knee"] == [None, "V", "V"]


def test_four_successful_runs_plus_one_failed_run():
    good_run = [
        {"rep": 1, "bottom": 2.5, "upper": "곧음", "knee": "정렬", "pause": "충분"},
    ]
    # 4 successful runs + 1 failed empty run
    aligned = align_reps_across_runs(
        [good_run, [], good_run, good_run, good_run], view="front"
    )
    assert len(aligned) == 1
    assert aligned[0]["successful_runs"] == 4
    assert aligned[0]["votes"]["upper"] == ["S", "S", "S", "S"]

    voted = vote_aligned_reps(aligned, view="front", min_successful_runs=3)
    cells = voted[0]["cells"]
    assert cells["upper"]["status"] == "confirmed"
    assert cells["upper"]["disputed"] is False
    assert cells["knee"]["status"] == "confirmed"
    assert cells["tempo"]["status"] == "confirmed"
    assert voted[0]["disputed_axes"] == []

    # Also verify via screen_both_clips when 1 of 5 Flash calls raises an error
    call_counter = {"front": 0, "side": 0}

    def fake_flash(norm_bytes, view, timeout_seconds=30.0):
        call_counter[view] += 1
        if view == "front" and call_counter["front"] == 1:
            raise RuntimeError("Transient Flash error")
        if view == "front":
            return {"reps": list(good_run)}
        return {
            "reps": [
                {"rep": 1, "bottom": 2.5, "upper": "곧음", "depth": "깊음", "pause": "충분"}
            ]
        }

    with patch("pipeline.call_flash_once", side_effect=fake_flash):
        screened = screen_both_clips(b"front", b"side", runs_per_clip=5)
    assert call_counter["front"] == 5  # 4 succeeded >= 3, so no retry needed
    assert screened["front"][0]["cells"]["upper"]["status"] == "confirmed"
    assert screened["front"][0]["disputed_axes"] == []


def test_fewer_than_three_successful_runs_retries_and_marks_disputed():
    good_side_rep = [
        {"rep": 1, "bottom": 2.5, "upper": "곧음", "depth": "깊음", "pause": "충분"}
    ]
    front_calls = {"n": 0}

    def mostly_failing_flash(norm_bytes, view, timeout_seconds=30.0):
        if view == "side":
            return {"reps": list(good_side_rep)}
        front_calls["n"] += 1
        # Only 2 succeed across initial 5 + 3 retries
        if front_calls["n"] in (1, 2):
            return {
                "reps": [
                    {"rep": 1, "bottom": 2.5, "upper": "곧음", "knee": "정렬", "pause": "충분"}
                ]
            }
        raise RuntimeError("Flash failure")

    with patch("pipeline.call_flash_once", side_effect=mostly_failing_flash):
        screened = screen_both_clips(b"front", b"side", runs_per_clip=5)

    # Initial 5 calls had 2 successes (<3) and 3 failures -> retried the 3 failed runs once (total 8)
    assert front_calls["n"] == 8
    # Since still only 2 runs succeeded (<3), every cell on front is marked disputed
    assert set(screened["front"][0]["disputed_axes"]) == {"upper", "knee", "tempo"}
    for ax in ("upper", "knee", "tempo"):
        assert screened["front"][0]["cells"][ax]["disputed"] is True
        assert screened["front"][0]["cells"][ax]["status"] == "verifying"


def test_majority_vote_tie_breaks_to_problem():
    assert majority_vote(["S", "S", "R", "R", None], "upper") == "R"
    assert majority_vote(["A", "A", "A", "V", "V"], "knee") == "A"
    assert majority_vote([None, None, None, None, None], "depth") == "H"


def test_vote_aligned_reps_confirmed_vs_disputed():
    aligned = [
        {
            "rep": 1,
            "bottom": 2.5,
            "successful_runs": 5,
            "votes": {
                "upper": ["S", "S", "S", "S", "S"],
                "knee": ["A", "V", "V", "V", "A"],
                "tempo": ["C", "C", "C", "C", None],
            },
        }
    ]
    voted = vote_aligned_reps(aligned, view="front")
    cells = voted[0]["cells"]
    assert cells["upper"]["status"] == "confirmed"
    assert cells["upper"]["disputed"] is False
    assert cells["upper"]["value"] == "S"

    assert cells["knee"]["status"] == "verifying"
    assert cells["knee"]["disputed"] is True
    assert cells["knee"]["value"] == "V"

    assert cells["tempo"]["status"] == "verifying"
    assert cells["tempo"]["disputed"] is True
    assert cells["tempo"]["value"] == "C"
    assert voted[0]["disputed_axes"] == ["knee", "tempo"]


def _make_voted_rep(rep_num: int, values: dict) -> dict:
    return {
        "rep": rep_num,
        "bottom": float(rep_num * 4),
        "cells": {k: {"value": v, "status": "confirmed"} for k, v in values.items()},
        "disputed_axes": [],
    }


def test_compute_squat_type():
    front_sadc = [
        _make_voted_rep(i, {"upper": "S", "knee": "A", "tempo": "C"})
        for i in range(1, 6)
    ]
    side_sadc = [
        _make_voted_rep(i, {"upper": "S", "depth": "D", "tempo": "C"})
        for i in range(1, 6)
    ]
    assert compute_squat_type(front_sadc, side_sadc) == "SADC"

    front_rvdb = [
        _make_voted_rep(1, {"upper": "R", "knee": "V", "tempo": "C"}),
        _make_voted_rep(2, {"upper": "R", "knee": "V", "tempo": "C"}),
        _make_voted_rep(3, {"upper": "R", "knee": "V", "tempo": "C"}),
        _make_voted_rep(4, {"upper": "S", "knee": "A", "tempo": "C"}),
        _make_voted_rep(5, {"upper": "S", "knee": "A", "tempo": "C"}),
    ]
    side_rvdb = [
        _make_voted_rep(1, {"upper": "S", "depth": "D", "tempo": "B"}),
        _make_voted_rep(2, {"upper": "S", "depth": "D", "tempo": "B"}),
        _make_voted_rep(3, {"upper": "S", "depth": "D", "tempo": "B"}),
        _make_voted_rep(4, {"upper": "S", "depth": "D", "tempo": "C"}),
        _make_voted_rep(5, {"upper": "S", "depth": "D", "tempo": "C"}),
    ]
    assert compute_squat_type(front_rvdb, side_rvdb) == "RVDB"


def test_select_recommendations_rules():
    res_pain = select_recommendations("RVDB", "Yes", "Regular", CONTENT)
    assert res_pain["pain_message"] == CONTENT["messages"]["pain"]
    assert res_pain["items"] == []

    front_reps = [
        _make_voted_rep(2, {"upper": "R", "knee": "V", "tempo": "B"}),
        _make_voted_rep(4, {"upper": "S", "knee": "V", "tempo": "C"}),
    ]
    res_rvdb = select_recommendations(
        "RVDB", "No", "Regular", CONTENT, front_reps=front_reps, side_reps=[]
    )
    assert res_rvdb["level"] == "Sometimes"
    assert [item["letter"] for item in res_rvdb["items"]] == ["V", "R"]
    assert res_rvdb["items"][0]["name"] == "Band squat"
    assert res_rvdb["items"][0]["why"] == "Why: knees caved in on reps 2 and 4."

    res_clear = select_recommendations("SADC", "No", "Regular", CONTENT)
    assert res_clear["level"] == "Regular"
    assert len(res_clear["items"]) == 1
    assert res_clear["items"][0]["name"] == "Bulgarian split squat"


def test_human_in_the_loop_scene_check_and_provisional():
    assert compute_scene_seek_seconds(0.4) == 0
    assert compute_scene_seek_seconds(8.9) == 7

    front = [
        {
            "rep": 1,
            "bottom": 3.5,
            "cells": {
                "upper": {"value": "S", "status": "confirmed", "user_checked": None},
                "knee": {"value": "V", "status": "verified", "user_checked": None},
                "tempo": {"value": "C", "status": "check", "user_checked": None},
            },
            "disputed_axes": ["knee", "tempo"],
        }
    ]
    side = [
        {
            "rep": 1,
            "bottom": 3.5,
            "cells": {
                "upper": {"value": "S", "status": "confirmed", "user_checked": None},
                "depth": {"value": "D", "status": "confirmed", "user_checked": None},
                "tempo": {"value": "C", "status": "confirmed", "user_checked": None},
            },
            "disputed_axes": [],
        }
    ]

    assert is_cell_flagged(front[0]["cells"]["upper"], "upper") is False
    assert is_cell_flagged(front[0]["cells"]["knee"], "knee") is True
    assert is_cell_flagged(front[0]["cells"]["tempo"], "tempo") is True
    assert count_pending_checks(front, side) == 2
    assert format_provisional_line(2) == "Provisional until you check 2 scenes."

    # User says 'Not quite' on knee -> flips V to A, pending drops to 1
    apply_user_scene_check(front[0], "knee", "not_quite")
    assert front[0]["cells"]["knee"]["value"] == "A"
    assert count_pending_checks(front, side) == 1
    assert format_provisional_line(1) == "Provisional until you check 1 scene."
    assert compute_squat_type(front, side) == "SADC"

    # User says 'Looks right' on tempo ('Please check this scene') -> confirmed, pending drops to 0
    apply_user_scene_check(front[0], "tempo", "looks_right")
    assert front[0]["cells"]["tempo"]["status"] == "confirmed"
    assert count_pending_checks(front, side) == 0
    assert format_provisional_line(0) is None


def test_grouped_scene_checks_and_provisional_count():
    assert format_rep_span([1, 2, 3, 4, 5]) == "reps 1–5"
    assert format_rep_span([2, 4]) == "reps 2 and 4"
    assert format_rep_span([3]) == "rep 3"

    # 5 front reps: knees caved in on reps 1-5 (rep 3 is verified by Argon, others confirmed),
    # plus 2 individual unclear ("check") cells on tempo (reps 2 and 5).
    front = []
    for r in range(1, 6):
        front.append(
            {
                "rep": r,
                "bottom": r * 4.0,
                "cells": {
                    "upper": {"value": "R", "status": "confirmed", "user_checked": None},
                    "knee": {
                        "value": "V",
                        "status": "verified" if r == 3 else "confirmed",
                        "user_checked": None,
                    },
                    "tempo": {
                        "value": "C" if r in (2, 5) else "B",
                        "status": "check" if r in (2, 5) else "confirmed",
                        "user_checked": None,
                    },
                },
                "disputed_axes": [],
            }
        )
    side = [
        {
            "rep": r,
            "bottom": r * 4.0,
            "cells": {
                "upper": {"value": "R", "status": "confirmed", "user_checked": None},
                "depth": {"value": "D", "status": "confirmed", "user_checked": None},
                "tempo": {"value": "B", "status": "confirmed", "user_checked": None},
            },
            "disputed_axes": [],
        }
        for r in range(1, 6)
    ]

    front_items = build_scene_check_items(front, "front")
    # Expect: 3 groups (upper, knee, tempo) + 2 individual "check" cells on tempo (reps 2 and 5) = 5 items on front
    group_items = [item for item in front_items if item["kind"] == "group"]
    indiv_items = [item for item in front_items if item["kind"] == "individual"]
    assert len(group_items) == 3
    assert len(indiv_items) == 2

    knee_group = next(item for item in group_items if item["axis"] == "knee")
    assert knee_group["label"] == "Front clip · Knees caved in on reps 1–5"
    # Representative rep prefers the one verified by Argon (rep 3, bottom 12.0s)
    assert knee_group["representative_rep"] == 3
    assert knee_group["bottom"] == 12.0
    assert knee_group["seek_seconds"] == 11

    # Side has 2 problem groups (upper R, tempo B) and 0 individual unclear cells
    side_items = build_scene_check_items(side, "side")
    assert len(side_items) == 2

    # Total provisional count = 3 front groups + 2 front individual + 2 side groups = 7
    assert count_pending_checks(front, side) == 7

    # Clicking "Not quite" on the knee group flips all 5 knee cells to "A" and recomputes type to RADB
    apply_group_scene_check(front, "knee", "not_quite")
    assert all(rep["cells"]["knee"]["value"] == "A" for rep in front)
    assert compute_squat_type(front, side) == "RADB"
    assert count_pending_checks(front, side) == 6

    # Clicking "Looks right" on the front upper group confirms all 5 upper cells
    apply_group_scene_check(front, "upper", "looks_right")
    assert all(rep["cells"]["upper"]["user_checked"] == "looks_right" for rep in front)
    assert count_pending_checks(front, side) == 5

