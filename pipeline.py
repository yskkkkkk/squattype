import base64
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
import json
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from google import genai
import imageio_ffmpeg

from logic import (
    align_reps_across_runs,
    apply_argon_result,
    build_recheck_prompt,
    vote_aligned_reps,
)

MAX_VIDEO_BYTES = 30 * 1024 * 1024
MAX_DURATION_SECONDS = 30.0
ARGON_TIMEOUT_SECONDS = 25.0

PROMPT_FRONT = """너는 맨몸 스쿼트 자세를 판정하는 코치야.
첨부한 영상은 내가 정면에서 찍은 맨몸 스쿼트 영상이야. 반복은 여러 번 있어.
반복을 나누고, 반복마다 가장 낮은 지점을 기준으로 아래 4가지를 판정해.

1. 상체 (upper)
- 곧음: 시선이 정면 또는 2~3m 앞 바닥을 향하고, 가슴이 펴져 있음
- 말림: 고개를 숙여 발 바로 앞을 보거나, 등 위쪽이 둥글게 말림

2. 무릎 (knee)
- 정렬: 무릎이 두 번째 발가락 방향이거나 그보다 바깥을 향함
- 안쪽: 내려가거나 올라오는 중에 두 무릎 사이 간격이 좁아지거나, 무릎이 엄지발가락 쪽이나 그 안쪽으로 모임

3. 깊이 (depth)
- 정면 영상에서는 판단할 수 없으므로 항상 "판단 불가"로 써

4. 바닥 정지 (pause)
- 충분: 가장 낮은 지점에서 1초 이상 멈춤
- 짧음: 멈췄지만 1초 미만
- 없음: 멈추지 않고 바로 올라옴

규칙:
- 바르게 한 반복은 바르다고 판정해. 문제를 억지로 찾지 마.
- 확신이 낮은 항목은 low_confidence 배열에 항목 이름(upper, knee, pause)을 넣어.
- bottom은 가장 낮은 지점에 도달한 시각(초, 소수점 한 자리)이야.

아래 JSON만 출력해. 다른 말은 쓰지 마.
{"reps":[{"rep":1,"bottom":0.0,"upper":"곧음|말림","knee":"정렬|안쪽","depth":"판단 불가","pause":"충분|짧음|없음","low_confidence":[]}]}"""

PROMPT_SIDE = """너는 맨몸 스쿼트 자세를 판정하는 코치야.
첨부한 영상은 내가 측면에서 찍은 맨몸 스쿼트 영상이야. 반복은 여러 번 있어.
반복을 나누고, 반복마다 가장 낮은 지점을 기준으로 아래 4가지를 판정해.

1. 상체 (upper)
- 곧음: 시선이 정면 또는 2~3m 앞 바닥을 향하고, 가슴이 펴져 있음
- 말림: 고개를 숙여 발 바로 앞을 보거나, 등 위쪽이 둥글게 말림

2. 무릎 (knee)
- 측면 영상에서는 판단할 수 없으므로 항상 "판단 불가"로 써

3. 깊이 (depth)
- 깊음: 고관절이 무릎 윗부분과 같은 높이이거나 더 낮음 (허벅지가 바닥과 평행하거나 그 이하)
- 얕음: 고관절이 무릎보다 확실히 높음

4. 바닥 정지 (pause)
- 충분: 가장 낮은 지점에서 1초 이상 멈춤
- 짧음: 멈췄지만 1초 미만
- 없음: 멈추지 않고 바로 올라옴

규칙:
- 바르게 한 반복은 바르다고 판정해. 문제를 억지로 찾지 마.
- 확신이 낮은 항목은 low_confidence 배열에 항목 이름(upper, depth, pause)을 넣어.
- bottom은 가장 낮은 지점에 도달한 시각(초, 소수점 한 자리)이야.

아래 JSON만 출력해. 다른 말은 쓰지 마.
{"reps":[{"rep":1,"bottom":0.0,"upper":"곧음|말림","knee":"판단 불가","depth":"깊음|얕음","pause":"충분|짧음|없음","low_confidence":[]}]}"""


class ClipTooLongError(ValueError):
    """Raised when an uploaded clip exceeds 30 seconds or 30 MB."""


class ModelParseError(RuntimeError):
    """Raised when the model response cannot be parsed as valid JSON."""

    def __init__(self, view: str, message: str = "") -> None:
        self.view = view
        super().__init__(message or f"Couldn't read the result for the {view} clip. Try again.")


def _probe_duration_seconds(src_path: str, ffmpeg_exe: str) -> float:
    """Read video duration in seconds from ffmpeg probe output or frame count."""
    proc = subprocess.run(
        [ffmpeg_exe, "-i", src_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        return hours * 3600 + minutes * 60 + seconds

    _, secs = imageio_ffmpeg.count_frames_and_secs(src_path)
    return float(secs or 0.0)


def normalize_clip(raw_bytes: bytes, suffix: str = ".mp4") -> Tuple[bytes, float]:
    """Validate size/duration and normalize clip to 720p H.264 mp4 in memory."""
    if len(raw_bytes) > MAX_VIDEO_BYTES:
        raise ClipTooLongError("Clip exceeds 30 MB limit.")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    src_fd, src_path = tempfile.mkstemp(suffix=suffix)
    dst_fd, dst_path = tempfile.mkstemp(suffix=".mp4")
    os.close(src_fd)
    os.close(dst_fd)

    try:
        with open(src_path, "wb") as f:
            f.write(raw_bytes)

        duration = _probe_duration_seconds(src_path, ffmpeg_exe)
        if duration > MAX_DURATION_SECONDS:
            raise ClipTooLongError(f"Clip duration {duration:.1f}s exceeds 30s limit.")

        cmd = [
            ffmpeg_exe,
            "-v",
            "error",
            "-y",
            "-i",
            src_path,
            "-an",
            "-vf",
            "scale='if(gt(iw,ih),-2,720)':'if(gt(iw,ih),720,-2)'",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            dst_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        with open(dst_path, "rb") as f:
            normalized_bytes = f.read()

        return normalized_bytes, duration
    finally:
        for path in (src_path, dst_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


def normalize_both_clips(
    front_raw: bytes,
    front_suffix: str,
    side_raw: bytes,
    side_suffix: str,
) -> Tuple[Tuple[bytes, float], Tuple[bytes, float]]:
    """Normalize front and side clips in parallel."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_front = pool.submit(normalize_clip, front_raw, front_suffix)
        fut_side = pool.submit(normalize_clip, side_raw, side_suffix)
        return fut_front.result(), fut_side.result()


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences surrounding JSON output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def get_genai_client(timeout_seconds: float = 30.0) -> genai.Client:
    """Create a Google GenAI client with an explicit request timeout in ms."""
    http_options = {"timeout": int(timeout_seconds * 1000)}
    return genai.Client(http_options=http_options)


def call_flash_once(
    normalized_bytes: bytes,
    view: str,
    timeout_seconds: float = 30.0,
) -> Dict[str, Any]:
    """Run one first-pass Gemini 3.8 Flash call on a normalized mp4 clip.

    Retries once silently if the model returns malformed JSON.
    """
    prompt = PROMPT_FRONT if view == "front" else PROMPT_SIDE
    b64 = base64.b64encode(normalized_bytes).decode("ascii")
    mime = "video/mp4"
    client = get_genai_client(timeout_seconds=timeout_seconds)

    last_error = None
    for _ in range(2):
        try:
            interaction = client.interactions.create(
                model="gemini-3.8-flash",
                input=[
                    {
                        "type": "video",
                        "data": b64,
                        "mime_type": mime,
                        "resolution": "high",
                        "processing": {"type": "static", "fps": 2},
                    },
                    {"type": "text", "text": prompt},
                ],
                generation_config={"thinking_level": "low"},
            )
            text = interaction.output_text or ""
            parsed = json.loads(strip_json_fences(text))
            if not isinstance(parsed, dict) or not isinstance(parsed.get("reps"), list):
                raise ValueError("Missing 'reps' list in model JSON.")
            return parsed
        except Exception as exc:
            last_error = exc

    raise ModelParseError(view) from last_error


def screen_both_clips(
    front_norm_bytes: bytes,
    side_norm_bytes: bytes,
    runs_per_clip: int = 5,
    timeout_seconds: float = 30.0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run 5 parallel Flash calls per clip (10 total) and align + vote reps.

    Drops failed/empty runs. If fewer than 3 runs succeed for a clip, retries
    the failed runs once.
    """
    front_runs: List[List[Dict[str, Any]]] = []
    side_runs: List[List[Dict[str, Any]]] = []

    with ThreadPoolExecutor(max_workers=runs_per_clip * 2) as pool:
        front_futs = [
            pool.submit(call_flash_once, front_norm_bytes, "front", timeout_seconds)
            for _ in range(runs_per_clip)
        ]
        side_futs = [
            pool.submit(call_flash_once, side_norm_bytes, "side", timeout_seconds)
            for _ in range(runs_per_clip)
        ]

        for fut in front_futs:
            try:
                reps = fut.result().get("reps", [])
                if reps:
                    front_runs.append(reps)
            except Exception:
                pass

        for fut in side_futs:
            try:
                reps = fut.result().get("reps", [])
                if reps:
                    side_runs.append(reps)
            except Exception:
                pass

    retry_front_count = (runs_per_clip - len(front_runs)) if len(front_runs) < 3 else 0
    retry_side_count = (runs_per_clip - len(side_runs)) if len(side_runs) < 3 else 0

    if retry_front_count > 0 or retry_side_count > 0:
        with ThreadPoolExecutor(max_workers=retry_front_count + retry_side_count) as pool:
            retry_front_futs = [
                pool.submit(call_flash_once, front_norm_bytes, "front", timeout_seconds)
                for _ in range(retry_front_count)
            ]
            retry_side_futs = [
                pool.submit(call_flash_once, side_norm_bytes, "side", timeout_seconds)
                for _ in range(retry_side_count)
            ]
            for fut in retry_front_futs:
                try:
                    reps = fut.result().get("reps", [])
                    if reps:
                        front_runs.append(reps)
                except Exception:
                    pass
            for fut in retry_side_futs:
                try:
                    reps = fut.result().get("reps", [])
                    if reps:
                        side_runs.append(reps)
                except Exception:
                    pass

    if not front_runs:
        raise ModelParseError("front")
    if not side_runs:
        raise ModelParseError("side")

    front_aligned = align_reps_across_runs(front_runs, view="front")
    side_aligned = align_reps_across_runs(side_runs, view="side")

    return {
        "front": vote_aligned_reps(
            front_aligned, view="front", expected_runs=len(front_runs)
        ),
        "side": vote_aligned_reps(
            side_aligned, view="side", expected_runs=len(side_runs)
        ),
    }


def call_argon_recheck_once(
    normalized_bytes: bytes,
    view: str,
    bottom: float,
    disputed_axes: Sequence[str],
    timeout_seconds: float = ARGON_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Verify one disputed rep window using Gemini 4 Argon at 5 fps.

    Worker-safe function: performs no UI calls and returns parsed JSON dict.
    """
    prompt = build_recheck_prompt(view, disputed_axes)
    b64 = base64.b64encode(normalized_bytes).decode("ascii")
    start_offset = f"{max(0.0, float(bottom) - 1.8):.1f}s"
    end_offset = f"{float(bottom) + 1.8:.1f}s"

    t_start = time.monotonic()
    last_error = None
    for attempt in range(2):
        remaining = timeout_seconds - (time.monotonic() - t_start)
        if remaining <= 1.0:
            break
        client = get_genai_client(timeout_seconds=remaining)
        try:
            interaction = client.interactions.create(
                model="gemini-4-argon",
                input=[
                    {
                        "type": "video",
                        "data": b64,
                        "mime_type": "video/mp4",
                        "resolution": "high",
                        "processing": {
                            "type": "static",
                            "fps": 5,
                            "start_offset": start_offset,
                            "end_offset": end_offset,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
                generation_config={"thinking_level": "low"},
            )
            text = interaction.output_text or ""
            parsed = json.loads(strip_json_fences(text))
            if not isinstance(parsed, dict):
                raise ValueError("Argon response is not a JSON object.")
            return parsed
        except Exception as exc:
            last_error = exc
            if attempt == 0 and (time.monotonic() - t_start) < timeout_seconds - 5.0:
                continue
            break

    raise RuntimeError("Argon verification failed or timed out.") from last_error


def recheck_disputed_reps_progressive(
    front_norm_bytes: bytes,
    side_norm_bytes: bytes,
    screened: Dict[str, List[Dict[str, Any]]],
    timeout_seconds: float = ARGON_TIMEOUT_SECONDS,
) -> Iterator[Dict[str, Any]]:
    """Run Argon rechecks in parallel with a strict 25s non-blocking wall timeout.

    Yields progress state dicts on the caller (main) thread as each future
    finishes or when the 25s deadline expires. Never uses a `with` block on
    ThreadPoolExecutor so slow threads cannot block exit.
    """
    tasks: List[Tuple[str, Dict[str, Any], bytes]] = []
    for view, clip_bytes in (("front", front_norm_bytes), ("side", side_norm_bytes)):
        for rep_entry in screened.get(view, []):
            if rep_entry.get("disputed_axes"):
                tasks.append((view, rep_entry, clip_bytes))

    if not tasks:
        return

    executor = ThreadPoolExecutor(max_workers=len(tasks))
    future_to_task = {}
    for view, rep_entry, clip_bytes in tasks:
        fut = executor.submit(
            call_argon_recheck_once,
            clip_bytes,
            view,
            rep_entry["bottom"],
            rep_entry["disputed_axes"],
            timeout_seconds,
        )
        future_to_task[fut] = (view, rep_entry)

    pending_futs = set(future_to_task.keys())
    done_labels: List[str] = []

    def _make_progress_payload() -> Dict[str, Any]:
        running_labels = [
            f"{future_to_task[f][0]} rep {future_to_task[f][1]['rep']}"
            for f in future_to_task
            if f in pending_futs
        ]
        return {
            "total": len(tasks),
            "completed": len(tasks) - len(pending_futs),
            "done_labels": list(done_labels),
            "running_labels": running_labels,
        }

    try:
        for fut in as_completed(future_to_task, timeout=timeout_seconds):
            pending_futs.discard(fut)
            view, rep_entry = future_to_task[fut]
            rep_num = rep_entry["rep"]
            try:
                raw_json: Optional[Dict[str, Any]] = fut.result(timeout=0)
            except Exception:
                raw_json = None

            apply_argon_result(rep_entry, raw_json)
            done_labels.append(f"{view} rep {rep_num} done")
            yield _make_progress_payload()
    except FuturesTimeoutError:
        pass
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    if pending_futs:
        for fut in list(pending_futs):
            pending_futs.discard(fut)
            view, rep_entry = future_to_task[fut]
            rep_num = rep_entry["rep"]
            apply_argon_result(rep_entry, None)
            done_labels.append(f"{view} rep {rep_num} done")
        yield _make_progress_payload()
