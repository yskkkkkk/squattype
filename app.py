import html
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import streamlit as st

from logic import (
    AXIS_CODES,
    VIEW_AXES,
    apply_user_scene_check,
    compute_scene_seek_seconds,
    compute_squat_type,
    count_pending_checks,
    format_provisional_line,
    is_cell_flagged,
    select_recommendations,
)
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
.sq-type-name {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1F2328;
    line-height: 1.25;
}
.sq-type-letters {
    font-family: monospace;
    font-size: 1.35rem;
    font-weight: 700;
    color: #0F766E;
    letter-spacing: 0.25em;
    margin-top: 0.25rem;
}
.sq-type-axes {
    font-size: 0.85rem;
    color: #475569;
    margin-top: 0.1rem;
}
.sq-type-line {
    font-size: 1rem;
    color: #1F2328;
    margin-top: 0.45rem;
}
.sq-provisional {
    font-size: 0.85rem;
    color: #475569;
    margin-top: 0.35rem;
}
</style>
""",
    unsafe_allow_html=True,
)


def get_value_word(axis: str, code: str) -> str:
    """Return the English label for a cell value code."""
    axis_cfg = CONTENT["axes"][axis]
    is_good = code == AXIS_CODES[axis]["good"]
    return axis_cfg["good"][1] if is_good else axis_cfg["problem"][1]


def format_cell_html(axis: str, cell: Dict[str, Any]) -> str:
    """Render a cell value + status label using DESIGN.md palette."""
    code = cell["value"]
    is_good = code == AXIS_CODES[axis]["good"]
    word = get_value_word(axis, code)
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
            if cells[ax].get("timeout_reason") and cells[ax].get("status") == "check":
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


def render_type_card_into_slot(
    slot: Any,
    screened: Dict[str, List[Dict[str, Any]]],
    survey_answers: Dict[str, str],
) -> str:
    """Render the SquatType card and optional sitting-rounded line into a slot."""
    squat_type = compute_squat_type(screened["front"], screened["side"])
    type_name, type_line = CONTENT["types"][squat_type]
    pending_count = count_pending_checks(screened["front"], screened["side"])
    provisional_text = format_provisional_line(pending_count)
    spaced_letters = "  ".join(list(squat_type))
    axis_labels = " · ".join(
        CONTENT["axes"][ax]["label"] for ax in ("upper", "knee", "depth", "tempo")
    )

    provisional_html = (
        f'<div class="sq-provisional">{html.escape(provisional_text)}</div>'
        if provisional_text
        else ""
    )

    with slot.container():
        with st.container(border=True):
            st.markdown(
                f'<div class="sq-type-name">{html.escape(type_name)}</div>'
                f'<div class="sq-type-letters">{html.escape(spaced_letters)}</div>'
                f'<div class="sq-type-axes">{html.escape(axis_labels)}</div>'
                f'<div class="sq-type-line">{html.escape(type_line)}</div>'
                f"{provisional_html}",
                unsafe_allow_html=True,
            )
        if survey_answers.get("day_posture") == "Sitting" and squat_type.startswith("R"):
            st.caption(CONTENT["messages"]["sitting_rounded"])

    return squat_type


def render_scene_check_buttons(
    reps: List[Dict[str, Any]],
    view: str,
) -> None:
    """Render a 'Check this scene' button for every flagged cell in a view."""
    axes = VIEW_AXES[view]
    for rep_entry in reps:
        rep_num = rep_entry["rep"]
        cells = rep_entry["cells"]
        for ax in axes:
            cell = cells[ax]
            if not is_cell_flagged(cell, ax):
                continue
            axis_label = CONTENT["axes"][ax]["label"]
            word = get_value_word(ax, cell["value"])
            checked_state = cell.get("user_checked")
            checked_note = ""
            if checked_state == "looks_right":
                checked_note = f" · {CONTENT['status']['looks_right']}"
            elif checked_state == "not_quite":
                checked_note = f" · {CONTENT['status']['not_quite']}"

            col_lbl, col_btn = st.columns([2, 1])
            with col_lbl:
                st.markdown(
                    f'<div class="sq-unclear" style="padding-top:0.4rem;">'
                    f"Rep {rep_num} · {html.escape(axis_label)} — {html.escape(word)}{html.escape(checked_note)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button(
                    CONTENT["status"]["check_button"],
                    key=f"check_btn_{view}_{rep_num}_{ax}",
                    use_container_width=True,
                ):
                    st.session_state["active_scene"] = {
                        "view": view,
                        "rep": rep_num,
                        "axis": ax,
                    }
                    st.session_state["active_scene_seek"] = compute_scene_seek_seconds(
                        rep_entry["bottom"]
                    )
                    st.rerun()


def render_active_scene_player_and_recommendations(
    screened: Dict[str, List[Dict[str, Any]]],
    front_norm: bytes,
    side_norm: bytes,
    survey_answers: Dict[str, str],
    squat_type: str,
) -> None:
    """Render the scene confirmation video player (if active) and recommendation cards."""
    active_scene: Optional[Dict[str, Any]] = st.session_state.get("active_scene")
    if active_scene:
        s_view = active_scene["view"]
        s_rep = active_scene["rep"]
        s_axis = active_scene["axis"]
        matching_rep = next(
            (r for r in screened.get(s_view, []) if r["rep"] == s_rep),
            None,
        )
        if matching_rep is not None:
            seek_s = compute_scene_seek_seconds(matching_rep["bottom"])
            st.session_state["active_scene_seek"] = seek_s
            clip_bytes = front_norm if s_view == "front" else side_norm
            axis_label = CONTENT["axes"][s_axis]["label"]
            word = get_value_word(s_axis, matching_rep["cells"][s_axis]["value"])

            st.caption(
                f"{s_view.capitalize()} clip · Rep {s_rep} · {axis_label} ({word}) — {seek_s} s"
            )
            st.video(clip_bytes, start_time=seek_s)

            col_q, col_yes, col_no = st.columns([2, 1, 1])
            with col_q:
                st.write("Is this right?")
            with col_yes:
                if st.button(
                    CONTENT["status"]["looks_right"],
                    key=f"scene_looks_right_{s_view}_{s_rep}_{s_axis}",
                    use_container_width=True,
                ):
                    apply_user_scene_check(matching_rep, s_axis, "looks_right")
                    st.rerun()
            with col_no:
                if st.button(
                    CONTENT["status"]["not_quite"],
                    key=f"scene_not_quite_{s_view}_{s_rep}_{s_axis}",
                    use_container_width=True,
                ):
                    apply_user_scene_check(matching_rep, s_axis, "not_quite")
                    st.rerun()

    recs = select_recommendations(
        squat_type=squat_type,
        pain=survey_answers.get("pain", "No"),
        experience=survey_answers.get("experience", "Beginner"),
        content=CONTENT,
        front_reps=screened["front"],
        side_reps=screened["side"],
    )

    st.markdown(
        f"**Try this next** <span class=\"sq-unclear\">({html.escape(CONTENT['recommendations']['label'])})</span>",
        unsafe_allow_html=True,
    )

    if recs["pain_message"]:
        with st.container(border=True):
            st.write(recs["pain_message"])
    else:
        for item in recs["items"]:
            with st.container(border=True):
                why_html = (
                    f'<div class="sq-unclear" style="margin-top:0.35rem;">{html.escape(item["why"])}</div>'
                    if item.get("why")
                    else ""
                )
                st.markdown(
                    f"<div><strong>{html.escape(item['name'])}</strong></div>"
                    f'<div style="margin-top:0.2rem;">{html.escape(item["line"])}</div>'
                    f"{why_html}",
                    unsafe_allow_html=True,
                )


st.title(CONTENT["app_name"])

st.subheader("1  About you")
survey_answers: Dict[str, str] = {}
for survey_item in CONTENT["survey"]:
    q_id = survey_item["id"]
    survey_answers[q_id] = st.radio(
        survey_item["question"],
        options=survey_item["options"],
        horizontal=True,
        key=f"survey_{q_id}",
    )

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
    st.session_state.pop("active_scene", None)
    st.session_state.pop("active_scene_seek", None)
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
            type_card_slot = st.empty()
            tab_front, tab_side = st.tabs(["Front clip", "Side clip"])
            front_table_slot = tab_front.empty()
            side_table_slot = tab_side.empty()

            render_type_card_into_slot(type_card_slot, screened, survey_answers)
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
                render_type_card_into_slot(type_card_slot, screened, survey_answers)
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

        with results_container:
            squat_type = render_type_card_into_slot(
                type_card_slot, screened, survey_answers
            )
            with tab_front:
                render_scene_check_buttons(screened["front"], "front")
            with tab_side:
                render_scene_check_buttons(screened["side"], "side")
            render_active_scene_player_and_recommendations(
                screened, front_norm, side_norm, survey_answers, squat_type
            )

    except ClipTooLongError:
        st.warning(CONTENT["messages"]["too_long"])
    except ModelParseError as err:
        st.warning(f"Couldn't read the result for the {err.view} clip. Try again.")
        if st.button("Try again"):
            st.rerun()

elif "screened" in st.session_state:
    screened_data = st.session_state["screened"]
    front_norm_data = st.session_state["front_norm"]
    side_norm_data = st.session_state["side_norm"]
    with results_container:
        st.subheader("3  Results")
        type_card_slot = st.empty()
        squat_type = render_type_card_into_slot(
            type_card_slot, screened_data, survey_answers
        )
        tab_front, tab_side = st.tabs(["Front clip", "Side clip"])
        with tab_front:
            st.markdown(
                build_view_table_html(screened_data["front"], "front"),
                unsafe_allow_html=True,
            )
            render_scene_check_buttons(screened_data["front"], "front")
        with tab_side:
            st.markdown(
                build_view_table_html(screened_data["side"], "side"),
                unsafe_allow_html=True,
            )
            render_scene_check_buttons(screened_data["side"], "side")
        render_active_scene_player_and_recommendations(
            screened_data, front_norm_data, side_norm_data, survey_answers, squat_type
        )

st.caption(CONTENT["messages"]["footer"])
