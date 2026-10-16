import json
from pathlib import Path
import time

import streamlit as st

from pipeline import ClipTooLongError, ModelParseError, call_flash_once, normalize_clip

CONTENT_PATH = Path(__file__).parent / "content.json"
with open(CONTENT_PATH, "r", encoding="utf-8") as f:
    CONTENT = json.load(f)

st.set_page_config(page_title=CONTENT["app_name"], layout="centered")

st.title(CONTENT["app_name"])

st.subheader("2  Your clips")
st.caption(CONTENT["messages"]["upload_guide"])

view_choice = st.radio(
    "Clip view",
    options=["Front clip", "Side clip"],
    horizontal=True,
)
selected_view = "front" if view_choice == "Front clip" else "side"

uploaded_clip = st.file_uploader(
    f"{view_choice} (mp4/mov)",
    type=["mp4", "mov"],
    key="m1_clip_uploader",
)

analyze_clicked = st.button(
    "Analyze my squat",
    type="primary",
    use_container_width=True,
    disabled=(uploaded_clip is None),
)

if analyze_clicked and uploaded_clip is not None:
    raw_bytes = uploaded_clip.getvalue()
    suffix = Path(uploaded_clip.name).suffix or ".mp4"

    try:
        with st.status(CONTENT["messages"]["preparing"], expanded=True) as status:
            t0 = time.monotonic()
            norm_bytes, duration = normalize_clip(raw_bytes, suffix=suffix)
            prep_elapsed = time.monotonic() - t0
            st.write(f"Preparing your clips… done in {prep_elapsed:.0f} s ({duration:.1f} s clip)")

            t1 = time.monotonic()
            st.write("Screening clip with Gemini 3.8 Flash…")
            result = call_flash_once(norm_bytes, view=selected_view)
            flash_elapsed = time.monotonic() - t1
            st.write(f"Screening clip with Gemini 3.8 Flash… done in {flash_elapsed:.0f} s")
            status.update(label="Analysis complete", state="complete", expanded=False)

        st.subheader("3  Results")
        st.video(norm_bytes, format="video/mp4")
        st.json(result)

    except ClipTooLongError:
        st.warning(CONTENT["messages"]["too_long"])
    except ModelParseError:
        st.warning(f"Couldn't read the result for the {selected_view} clip. Try again.")
        if st.button("Try again"):
            st.rerun()

st.caption(CONTENT["messages"]["footer"])
