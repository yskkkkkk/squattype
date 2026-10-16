import html
import json
from pathlib import Path
import time
from typing import Any, Dict, List

import streamlit as st

from logic import AXIS_CODES, VIEW_AXES
from pipeline import (
    ClipTooLongError,
    ModelParseError,
    normalize_both_clips,
    recheck_disputed_reps_progressive,
    screen_both_clips,
)

CONTENT_PATH = Path(__file__).parent / "content.json"
with open(CONTENT_PATH, "r", encoding="utf-8") as f:
    CONTENT = json.load(f)

st.set_page_config(page_title=CONTENT["app_name"], layout="centered")

st.markdown(
    """
<style>
.sq-table {
    width: 100%;
    border-collapse: collapse;
    background-color: #F4F4F2;
    font-size: 0.95rem;
    margin-bottom: 0.75rem;
}
.sq-table th, .sq-table td {
    padding: 0.55rem 0.75rem;
    text-align: left;
    border-bottom: 1px solid #FFFFFF;
}
.sq-table th {
    color: #1F2328;
    font-weight: 600;
}
.sq-good {
    color: #15803D;
    font-weight: 600;
}
.sq-problem {
    color: #B45309;
    font-weight: 600;
}
.sq-unclear {
    color: #475569;
    font-size: 0.85rem;
}
</style>
""",
    unsafe_allow_html=True,
)


def format_cell_html(axis: str, cell: Dict[str, Any]) -> str:
    """Render a cell value + status label using DESIGN.md palette."""
    axis_cfg = CONTENT["axes"][axis]
    code = cell["value"]
    is_good = code == AXIS_CODES[axis]["good"]
    word = axis_cfg["good"][1] if is_good else axis_cfg["problem"][1]
    css_class = "sq-good" if is_good else "sq-problem"
    status = cell.get("status", "confirmed")

    value_span = f'<span class="{css_class}">{html.escape(word)}</span>'
    if status == "verifying":
        status_text = html.escape(CONTENT["status"]["verifying"])
        return f'{value_span} <span class="sq-unclear">· {status_text}</span>'
    if status == "verified":
        status_text = html.escape(CONTENT["status"]["verified"])
        return f'{value_span} <span class="sq-unclear">· {status_text}</span>'
    if status == "check":
        status_text = html.escape(CONTENT["status"]["check"])
        return f'{value_span} <span class="sq-unclear">· {status_text}</span>'
    return value_span


def build_view_table_html(reps: List[Dict[str, Any]], view: str) -> str:
    """Build the HTML string for a rep x axis table and any timeout notes."""
    axes = VIEW_AXES[view]
    headers = ["Rep"] + [CONTENT["axes"][ax]["label"] for ax in axes]
    header_html = "".join(f"<th>{html.escape(h)}</th>" for h in headers)

    rows_html = []
    timeout_notes: List[str] = []
    for rep_item in reps:
        rep_num = rep_item["rep"]
        cells = rep_item["cells"]
        tds = [f"<td>{rep_num}</td>"]
        rep_timed_out = False
        for ax in axes:
            tds.append(f"<td>{format_cell_html(ax, cells[ax])}</td>")
            if cells[ax].get("timeout_reason"):
                rep_timed_out = True
        if rep_timed_out:
            timeout_notes.append(f"Argon took too long on rep {rep_num}.")
        rows_html.append(f"<tr>{''.join(tds)}</tr>")

    notes_html = "".join(
        f'<div class="sq-unclear">{html.escape(note)}</div>'
        for note in timeout_notes
    )
    return (
        f'<table class="sq-table">'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table>"
        f"{notes_html}"
    )


def format_argon_progress_line(
    total: int,
    done_labels: List[str],
    running_labels: List[str],
) -> str:
    """Format the live Argon verification line for st.status."""
    noun = "rep" if total == 1 else "reps"
    parts = list(done_labels)
    if running_labels:
        parts.append(", ".join(f"{lbl} running" for lbl in running_labels))
    detail = ", ".join(parts)
    if detail:
        return f"Verifying {total} {noun} with Gemini 4 Argon… {detail}"
    return f"Verifying {total} {noun} with Gemini 4 Argon…"


st.title(CONTENT["app_name"])

st.subheader("2  Your clips")
st.caption(CONTENT["messages"]["upload_guide"])

col_front, col_side = st.columns(2)
with col_front:
    front_upload = st.file_uploader(
        "Front clip (mp4/mov)",
        type=["mp4", "mov"],
        key="front_clip_uploader",
    )
with col_side:
    side_upload = st.file_uploader(
        "Side clip (mp4/mov)",
        type=["mp4", "mov"],
        key="side_clip_uploader",
    )

both_uploaded = (front_upload is not None) and (side_upload is not None)
analyze_clicked = st.button(
    "Analyze my squat",
    type="primary",
    use_container_width=True,
    disabled=not both_uploaded,
)

status_container = st.container()
results_container = st.container()

if analyze_clicked and both_uploaded:
    front_raw = front_upload.getvalue()
    front_suffix = Path(front_upload.name).suffix or ".mp4"
    side_raw = side_upload.getvalue()
    side_suffix = Path(side_upload.name).suffix or ".mp4"

    try:
        with status_container:
            status_box = st.status(CONTENT["messages"]["preparing"], expanded=True)
            prep_line = status_box.empty()
            flash_line = status_box.empty()
            argon_line = status_box.empty()

            prep_line.write(CONTENT["messages"]["preparing"])
            t0 = time.monotonic()
            (front_norm, _), (side_norm, _) = normalize_both_clips(
                front_raw, front_suffix, side_raw, side_suffix
            )
            prep_s = time.monotonic() - t0
            prep_line.write(f"Preparing your clips… done in {prep_s:.0f} s")

            status_box.update(
                label="Screening both clips with Gemini 3.8 Flash (10 runs)…",
                state="running",
            )
            flash_line.write("Screening both clips with Gemini 3.8 Flash (10 runs)…")
            t1 = time.monotonic()
            screened = screen_both_clips(front_norm, side_norm, runs_per_clip=5)
            flash_s = time.monotonic() - t1
            flash_line.write(
                f"Screening both clips with Gemini 3.8 Flash (10 runs)… done in {flash_s:.0f} s"
            )

        with results_container:
            st.subheader("3  Results")
            tab_front, tab_side = st.tabs(["Front clip", "Side clip"])
            front_table_slot = tab_front.empty()
            side_table_slot = tab_side.empty()

            front_table_slot.markdown(
                build_view_table_html(screened["front"], "front"),
                unsafe_allow_html=True,
            )
            side_table_slot.markdown(
                build_view_table_html(screened["side"], "side"),
                unsafe_allow_html=True,
            )

        disputed_reps = [
            (v, r)
            for v in ("front", "side")
            for r in screened[v]
            if r.get("disputed_axes")
        ]
        if disputed_reps:
            total_disputed = len(disputed_reps)
            noun = "rep" if total_disputed == 1 else "reps"
            status_box.update(
                label=f"Verifying {total_disputed} {noun} with Gemini 4 Argon…",
                state="running",
            )
            argon_line.write(
                format_argon_progress_line(
                    total_disputed,
                    [],
                    [f"{v} rep {r['rep']}" for v, r in disputed_reps],
                )
            )
            t2 = time.monotonic()
            for progress in recheck_disputed_reps_progressive(
                front_norm, side_norm, screened
            ):
                argon_line.write(
                    format_argon_progress_line(
                        progress["total"],
                        progress["done_labels"],
                        progress["running_labels"],
                    )
                )
                front_table_slot.markdown(
                    build_view_table_html(screened["front"], "front"),
                    unsafe_allow_html=True,
                )
                side_table_slot.markdown(
                    build_view_table_html(screened["side"], "side"),
                    unsafe_allow_html=True,
                )
            argon_s = time.monotonic() - t2
            argon_line.write(
                f"Verifying {total_disputed} {noun} with Gemini 4 Argon… done in {argon_s:.0f} s"
            )

        status_box.update(label="Analysis complete", state="complete", expanded=False)
        st.session_state["front_norm"] = front_norm
        st.session_state["side_norm"] = side_norm
        st.session_state["screened"] = screened

    except ClipTooLongError:
        st.warning(CONTENT["messages"]["too_long"])
    except ModelParseError as err:
        st.warning(f"Couldn't read the result for the {err.view} clip. Try again.")
        if st.button("Try again"):
            st.rerun()

elif "screened" in st.session_state:
    screened_data = st.session_state["screened"]
    with results_container:
        st.subheader("3  Results")
        tab_front, tab_side = st.tabs(["Front clip", "Side clip"])
        with tab_front:
            st.markdown(
                build_view_table_html(screened_data["front"], "front"),
                unsafe_allow_html=True,
            )
        with tab_side:
            st.markdown(
                build_view_table_html(screened_data["side"], "side"),
                unsafe_allow_html=True,
            )

st.caption(CONTENT["messages"]["footer"])
