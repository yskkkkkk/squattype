import html
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List

import streamlit as st

from logic import (
    AXIS_CODES,
    VIEW_AXES,
    apply_group_scene_check,
    apply_user_scene_check,
    build_scene_check_items,
    compute_squat_type,
    count_pending_checks,
    format_provisional_line,
    select_recommendations,
)
from pipeline import (
    ClipTooLongError,
    ModelParseError,
    compare_front_sessions,
    normalize_both_clips,
    normalize_clip,
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
        rep_timeout_reason = None
        for ax in axes:
            tds.append(f"<td>{format_cell_html(ax, cells[ax])}</td>")
            if cells[ax].get("timeout_reason") and cells[ax].get("status") == "check":
                rep_timeout_reason = cells[ax]["timeout_reason"]
        if rep_timeout_reason:
            timeout_notes.append(rep_timeout_reason)
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


def render_scene_checks_with_inline_player(
    reps: List[Dict[str, Any]],
    view: str,
    clip_bytes: bytes,
) -> None:
    """Render grouped problem rows and individual unclear rows with inline scene player."""
    items = build_scene_check_items(reps, view)
    view_title = "Front clip" if view == "front" else "Side clip"
    active_scene_id = st.session_state.get("active_scene_id")

    for item in items:
        item_id = item["id"]
        checked_state = item.get("user_checked")
        checked_note = ""
        if checked_state == "looks_right":
            checked_note = f" · {CONTENT['status']['looks_right']}"
        elif checked_state == "not_quite":
            checked_note = f" · {CONTENT['status']['not_quite']}"

        col_lbl, col_btn = st.columns([2, 1])
        with col_lbl:
            st.markdown(
                f'<div class="sq-unclear" style="padding-top:0.4rem;">'
                f"{html.escape(item['label'])}{html.escape(checked_note)}"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button(
                CONTENT["status"]["check_button"],
                key=f"check_btn_{item_id}",
                use_container_width=True,
            ):
                st.session_state["active_scene_id"] = item_id
                st.session_state["active_scene_seek"] = item["seek_seconds"]
                st.rerun()

        if active_scene_id == item_id:
            rep_num = item["representative_rep"]
            seek_s = item["seek_seconds"]
            st.caption(f"{view_title} · rep {rep_num} · from {seek_s} s")
            st.video(clip_bytes, start_time=seek_s, autoplay=True, muted=True)

            col_q, col_yes, col_no = st.columns([2, 1, 1])
            with col_q:
                st.write("Is this right?")
            with col_yes:
                if st.button(
                    CONTENT["status"]["looks_right"],
                    key=f"scene_looks_right_{item_id}",
                    use_container_width=True,
                ):
                    if item["kind"] == "group":
                        apply_group_scene_check(reps, item["axis"], "looks_right")
                    else:
                        matching_rep = next(
                            (r for r in reps if r["rep"] == rep_num), None
                        )
                        if matching_rep is not None:
                            apply_user_scene_check(
                                matching_rep, item["axis"], "looks_right"
                            )
                    st.session_state.pop("active_scene_id", None)
                    st.rerun()
            with col_no:
                if st.button(
                    CONTENT["status"]["not_quite"],
                    key=f"scene_not_quite_{item_id}",
                    use_container_width=True,
                ):
                    if item["kind"] == "group":
                        apply_group_scene_check(reps, item["axis"], "not_quite")
                    else:
                        matching_rep = next(
                            (r for r in reps if r["rep"] == rep_num), None
                        )
                        if matching_rep is not None:
                            apply_user_scene_check(
                                matching_rep, item["axis"], "not_quite"
                            )
                    st.session_state.pop("active_scene_id", None)
                    st.rerun()


def render_recommendations(
    screened: Dict[str, List[Dict[str, Any]]],
    survey_answers: Dict[str, str],
    squat_type: str,
) -> None:
    """Render the 'Try this next' exercise recommendation section."""
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

if (analyze_clicked or st.session_state.pop("retry_requested", False)) and both_uploaded:
    st.session_state.pop("active_scene_id", None)
    st.session_state.pop("active_scene_seek", None)
    st.session_state.pop("compare_result", None)
    front_raw = front_upload.getvalue()
    front_suffix = Path(front_upload.name).suffix or ".mp4"
    side_raw = side_upload.getvalue()
    side_suffix = Path(side_upload.name).suffix or ".mp4"
    tips_list: List[str] = CONTENT.get("tips", [])

    try:
        t_total_start = time.monotonic()
        status_lines: List[str] = []
        with status_container:
            status_box = st.status(CONTENT["messages"]["preparing"], expanded=True)
            prep_line = status_box.empty()
            flash_line = status_box.empty()
            argon_line = status_box.empty()
            tip_slot = st.empty()

            if tips_list:
                tip_slot.caption(f"Tip: {tips_list[0]}")

            prep_line.write(CONTENT["messages"]["preparing"])
            t0 = time.monotonic()
            (front_norm, _), (side_norm, _) = normalize_both_clips(
                front_raw, front_suffix, side_raw, side_suffix
            )
            prep_s = time.monotonic() - t0
            prep_done_msg = f"Preparing your clips… done in {prep_s:.0f} s"
            prep_line.write(prep_done_msg)
            status_lines.append(prep_done_msg)

            if len(tips_list) > 1:
                tip_slot.caption(f"Tip: {tips_list[1]}")

            status_box.update(
                label="Screening both clips with Gemini 3.8 Flash (10 runs)…",
                state="running",
            )
            flash_line.write("Screening both clips with Gemini 3.8 Flash (10 runs)…")
            t1 = time.monotonic()
            screened = screen_both_clips(front_norm, side_norm, runs_per_clip=5)
            flash_s = time.monotonic() - t1
            flash_done_msg = f"Screening done in {flash_s:.0f} s"
            flash_line.write(flash_done_msg)
            status_lines.append(flash_done_msg)

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
            if len(tips_list) > 2:
                tip_slot.caption(f"Tip: {tips_list[2]}")

            t2 = time.monotonic()
            tip_idx = 2
            for progress in recheck_disputed_reps_progressive(
                front_norm, side_norm, screened
            ):
                if tips_list:
                    tip_idx = (tip_idx + 1) % len(tips_list)
                    tip_slot.caption(f"Tip: {tips_list[tip_idx]}")
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
            argon_done_msg = (
                f"Verified {total_disputed} {noun} with Gemini 4 Argon in {argon_s:.0f} s"
            )
            argon_line.write(argon_done_msg)
            status_lines.append(argon_done_msg)

        tip_slot.empty()
        total_s = time.monotonic() - t_total_start
        final_status_label = f"Analysis complete in {total_s:.0f} s"
        status_box.update(
            label=final_status_label, state="complete", expanded=False
        )
        st.session_state["front_norm"] = front_norm
        st.session_state["side_norm"] = side_norm
        st.session_state["screened"] = screened
        st.session_state["status_label"] = final_status_label
        st.session_state["status_lines"] = status_lines

        with results_container:
            squat_type = render_type_card_into_slot(
                type_card_slot, screened, survey_answers
            )
            with tab_front:
                render_scene_checks_with_inline_player(
                    screened["front"], "front", front_norm
                )
            with tab_side:
                render_scene_checks_with_inline_player(
                    screened["side"], "side", side_norm
                )
            render_recommendations(screened, survey_answers, squat_type)

    except ClipTooLongError:
        st.warning(CONTENT["messages"]["too_long"])
    except ModelParseError as err:
        st.warning(f"Couldn't read the result for the {err.view} clip. Try again.")
        if st.button("Try again"):
            st.session_state["retry_requested"] = True
            st.rerun()
    except Exception:
        st.warning("Couldn't read this clip. Try a different file.")

elif "screened" in st.session_state:
    screened_data = st.session_state["screened"]
    front_norm_data = st.session_state["front_norm"]
    side_norm_data = st.session_state["side_norm"]
    with status_container:
        saved_label = st.session_state.get("status_label", "Analysis complete")
        with st.status(saved_label, state="complete", expanded=False):
            for line in st.session_state.get("status_lines", []):
                st.write(line)
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
            render_scene_checks_with_inline_player(
                screened_data["front"], "front", front_norm_data
            )
        with tab_side:
            st.markdown(
                build_view_table_html(screened_data["side"], "side"),
                unsafe_allow_html=True,
            )
            render_scene_checks_with_inline_player(
                screened_data["side"], "side", side_norm_data
            )
        render_recommendations(screened_data, survey_answers, squat_type)

st.subheader("4  Compare sessions")
prev_front_upload = st.file_uploader(
    "Previous front clip (mp4/mov)",
    type=["mp4", "mov"],
    key="prev_front_clip_uploader",
)
can_compare = (prev_front_upload is not None) and ("front_norm" in st.session_state)
compare_clicked = st.button(
    CONTENT["messages"]["compare_button"],
    use_container_width=True,
    disabled=not can_compare,
)

if compare_clicked and can_compare:
    prev_raw = prev_front_upload.getvalue()
    prev_suffix = Path(prev_front_upload.name).suffix or ".mp4"
    try:
        with st.spinner("Comparing sessions with Gemini 4 Argon…"):
            prev_norm, _ = normalize_clip(prev_raw, suffix=prev_suffix)
            comparison_text = compare_front_sessions(
                prev_norm, st.session_state["front_norm"]
            )
            st.session_state["compare_result"] = comparison_text
    except ClipTooLongError:
        st.warning(CONTENT["messages"]["too_long"])
    except Exception:
        st.warning("Couldn't compare the two sessions. Try again.")

if st.session_state.get("compare_result"):
    st.write(st.session_state["compare_result"])

with st.expander("How it works", expanded=False):
    how_sentences = [
        s.strip()
        for s in re.split(r"(?<=\.)\s+", CONTENT["messages"]["how_it_works"])
        if s.strip()
    ]
    for idx, sentence in enumerate(how_sentences, start=1):
        st.write(f"{idx}  {sentence}")

st.caption(CONTENT["messages"]["footer"])
