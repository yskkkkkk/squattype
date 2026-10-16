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
    margin-bottom: 1rem;
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


def render_view_table(reps: List[Dict[str, Any]], view: str) -> None:
    """Render a rep x axis table for a single view."""
    axes = VIEW_AXES[view]
    headers = ["Rep"] + [CONTENT["axes"][ax]["label"] for ax in axes]
    header_html = "".join(f"<th>{html.escape(h)}</th>" for h in headers)

    rows_html = []
    for rep_item in reps:
        rep_num = rep_item["rep"]
        cells = rep_item["cells"]
        tds = [f"<td>{rep_num}</td>"]
        for ax in axes:
            tds.append(f"<td>{format_cell_html(ax, cells[ax])}</td>")
        rows_html.append(f"<tr>{''.join(tds)}</tr>")

    table_html = (
        f'<table class="sq-table">'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


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

if analyze_clicked and both_uploaded:
    front_raw = front_upload.getvalue()
    front_suffix = Path(front_upload.name).suffix or ".mp4"
    side_raw = side_upload.getvalue()
    side_suffix = Path(side_upload.name).suffix or ".mp4"

    try:
        with st.status(CONTENT["messages"]["preparing"], expanded=True) as status_box:
            t0 = time.monotonic()
            (front_norm, _), (side_norm, _) = normalize_both_clips(
                front_raw, front_suffix, side_raw, side_suffix
            )
            prep_s = time.monotonic() - t0
            st.write(f"Preparing your clips… done in {prep_s:.0f} s")

            t1 = time.monotonic()
            status_box.update(
                label="Screening both clips with Gemini 3.8 Flash (10 runs)…",
                state="running",
            )
            screened = screen_both_clips(front_norm, side_norm, runs_per_clip=5)
            flash_s = time.monotonic() - t1
            st.write(
                f"Screening both clips with Gemini 3.8 Flash (10 runs)… done in {flash_s:.0f} s"
            )
            status_box.update(label="Screening complete", state="complete", expanded=False)

        st.session_state["front_norm"] = front_norm
        st.session_state["side_norm"] = side_norm
        st.session_state["screened"] = screened

    except ClipTooLongError:
        st.warning(CONTENT["messages"]["too_long"])
    except ModelParseError as err:
        st.warning(f"Couldn't read the result for the {err.view} clip. Try again.")
        if st.button("Try again"):
            st.rerun()

if "screened" in st.session_state:
    screened_data = st.session_state["screened"]
    st.subheader("3  Results")

    tab_front, tab_side = st.tabs(["Front clip", "Side clip"])
    with tab_front:
        render_view_table(screened_data["front"], "front")
    with tab_side:
        render_view_table(screened_data["side"], "side")

st.caption(CONTENT["messages"]["footer"])
