import base64
import json
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, Tuple

from google import genai
import imageio_ffmpeg

MAX_VIDEO_BYTES = 30 * 1024 * 1024
MAX_DURATION_SECONDS = 30.0

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


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences surrounding JSON output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def get_genai_client(timeout_seconds: float = 30.0) -> genai.Client:
    """Create a Google GenAI client using GEMINI_API_KEY from environment."""
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
            if not isinstance(parsed, dict) or "reps" not in parsed:
                raise ValueError("Missing 'reps' key in model JSON.")
            return parsed
        except Exception as exc:
            last_error = exc

    raise ModelParseError(f"Couldn't read the result for the {view} clip.") from last_error
