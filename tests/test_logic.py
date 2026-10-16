import json
from pathlib import Path
import time
from unittest.mock import patch

from logic import (
    align_reps_across_runs,
    apply_argon_result,
    build_recheck_prompt,
    compute_squat_type,
    majority_vote,
    normalize_korean_value,
    select_recommendations,
    vote_aligned_reps,
)
from pipeline import recheck_disputed_reps_progressive

CONTENT_PATH = Path(__file__).parent.parent / "content.json"
with open(CONTENT_PATH, "r", encoding="utf-8") as f:
    CONTENT = json.load(f)


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
            "votes": {
                "upper": ["S", "S", "S", "S", "S"],
                "knee": ["V", "V", "V", "A", "A"],
                "tempo": ["B", "B", "B", "B", "B"],
            },
        }
    ]
    voted = vote_aligned_reps(aligned, view="front", expected_runs=5)
    apply_argon_result(voted[0], {"knee": "안쪽(두 무릎 사이 간격이 좁아지거나)"})
    assert voted[0]["cells"]["knee"]["value"] == "V"
    assert voted[0]["cells"]["knee"]["status"] == "verified"
    assert voted[0]["cells"]["knee"]["timeout_reason"] is None

    # Timeout fallback uses majority_value and sets status='check'
    voted_timeout = vote_aligned_reps(aligned, view="front", expected_runs=5)
    apply_argon_result(voted_timeout[0], None)
    assert voted_timeout[0]["cells"]["knee"]["value"] == "V"
    assert voted_timeout[0]["cells"]["knee"]["status"] == "check"
    assert voted_timeout[0]["cells"]["knee"]["timeout_reason"] == "Argon took too long on rep 2."


def test_recheck_timeout_does_not_wait_for_slow_thread():
    aligned = [
        {
            "rep": 3,
            "bottom": 12.0,
            "votes": {
                "upper": ["S", "S", "S", "S", "S"],
                "depth": ["D", "H", "H", "D", "H"],
                "tempo": ["C", "C", "C", "C", "C"],
            },
        }
    ]
    screened = {
        "front": [],
        "side": vote_aligned_reps(aligned, view="side", expected_runs=5),
    }

    def slow_argon(*args, **kwargs):
        time.sleep(5.0)
        return {"depth": "깊음"}

    t0 = time.monotonic()
    with patch("pipeline.call_argon_recheck_once", side_effect=slow_argon):
        events = list(
            recheck_disputed_reps_progressive(
                b"front", b"side", screened, timeout_seconds=0.2
            )
        )
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0
    assert len(events) == 1
    cell = screened["side"][0]["cells"]["depth"]
    assert cell["status"] == "check"
    assert cell["value"] == "H"
    assert cell["timeout_reason"] == "Argon took too long on rep 3."


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


def test_majority_vote_tie_breaks_to_problem():
    assert majority_vote(["S", "S", "R", "R", None], "upper") == "R"
    assert majority_vote(["A", "A", "A", "V", "V"], "knee") == "A"
    assert majority_vote([None, None, None, None, None], "depth") == "H"


def test_vote_aligned_reps_confirmed_vs_disputed():
    aligned = [
        {
            "rep": 1,
            "bottom": 2.5,
            "votes": {
                "upper": ["S", "S", "S", "S", "S"],
                "knee": ["A", "V", "V", "V", "A"],
                "tempo": ["C", "C", "C", "C", None],
            },
        }
    ]
    voted = vote_aligned_reps(aligned, view="front", expected_runs=5)
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
