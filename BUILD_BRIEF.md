# BUILD_BRIEF — SquatType

Read this whole file and DESIGN.md before writing code. Then reply with (1) your questions and (2) a build plan by milestone. Do not start coding until the plan is agreed.

Everything below was decided and validated in API experiments before the event. Follow it exactly; do not substitute other models, parameters or SDKs.

## 1. What we are building

A web app that gives squat form feedback from two phone videos.

- **User:** someone who learns squats from videos and has no one to check their form.
- **Track:** Google AI for social good (health).
- **Flow:** short survey → upload a front clip and a side clip → per-rep judgment on 4 axes → one of 16 squat types → exercise recommendations → user confirms flagged scenes → optional "compare with my last session".
- **Core idea:** Gemini 3.8 Flash screens, Gemini 4 Argon verifies only the disputed reps, the user confirms anything still unclear. "AI looks, code decides": the model only reports observations; votes, types and recommendations are computed in code.

## 2. Stack and hard rules

- Python 3.11, Streamlit, `google-genai` >= 2.29 (Interactions API: `client.interactions.create`). No other LLM SDK.
- Deploy: Cloud Run, region `asia-northeast3`, HTTPS only. API key from Secret Manager as env var `GEMINI_API_KEY`.
- **Never** put an API key in code, config, logs or git history. The repo is public. `.gitignore` must be in the first commit and include `.env`, `*.key`, `key.txt`, `*.mp4`, `*.mov`, `test_videos/`.
- **Normalize every uploaded clip before any API call** (section 3a). All validation used normalized clips.
- Upload limits: guide users to about 20 seconds; reject only clips longer than 30 seconds or larger than 30 MB (the test clips are 16–27 s).
- **Never** store uploaded videos. Keep bytes in memory for the request only. Do not log video bytes or base64.
- **Never** commit test videos (they show a real face). Tests read them from a folder outside the repo given by env var `TEST_VIDEO_DIR`.
- Always pass `generation_config={"thinking_level": "low"}` explicitly. Argon's default is "medium", which is slower and did not improve accuracy. Do not pass temperature/top_p/top_k.
- Wording rule: this is **form feedback, not medical advice**. Never use the words diagnosis, injury risk, rehab, treatment, therapy (except "physical therapist" in the pain message).
- UI language: English. Display strings come from `content.json` (provided). Prompts stay in Korean exactly as written below (they were validated in Korean); map the Korean output values to codes in code.

## 3. Models and exact call shapes

| Job | Model ID |
|---|---|
| First-pass screening (5 parallel runs per clip) | `gemini-3.8-flash` |
| Verification of disputed reps, session comparison | `gemini-4-argon` |

First pass (whole clip):
```python
interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input=[
        {"type": "video", "data": b64, "mime_type": mime, "resolution": "high",
         "processing": {"type": "static", "fps": 2}},
        {"type": "text", "text": PROMPT_FRONT_OR_SIDE},
    ],
    generation_config={"thinking_level": "low"},
)
text = interaction.output_text
```

Verification of one rep (clip window, higher fps):
```python
{"type": "video", "data": b64, "mime_type": mime, "resolution": "high",
 "processing": {"type": "static", "fps": 5,
                "start_offset": f"{max(0, bottom - 1.8):.1f}s",
                "end_offset": f"{bottom + 1.8:.1f}s"}}
# model="gemini-4-argon", thinking_level "low", prompt = RECHECK prompt below
```

Session comparison (two clips in one request): two video parts (previous first, today second), fps 2, resolution high, then the COMPARE prompt; model `gemini-4-argon`, thinking_level "low".

### 3a. Input normalization (required)

Run this on each uploaded clip (both clips in parallel) and use the result for the API calls **and** for the on-screen player, so timestamps match:
```python
import imageio_ffmpeg, subprocess, tempfile
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
# shorter side 720 px, rotation metadata applied (ffmpeg autorotates by default), H.264, no audio
cmd = [FFMPEG, "-v", "error", "-y", "-i", src_path, "-an",
       "-vf", "scale='if(gt(iw,ih),-2,720)':'if(gt(iw,ih),720,-2)'",
       "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
       "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst_path]
```
- Write the upload to a temp file, convert, read the result into memory, delete both temp files.
- Read the duration (e.g. from ffmpeg's stderr or `imageio_ffmpeg.count_frames_and_secs`) and reject clips over 30 s before converting.
- Takes about 4–5 s for a 25 s phone clip. Output is 1–4 MB, always `video/mp4`.
- Why: the same raw iPhone clip (HEVC, 1080p, rotation metadata) got "knees caving in" right 1 time in 3; the normalized 720p H.264 version got it right 3 times in 3.

Notes from experiments:
- `fps=2` with `resolution="high"` is the validated setting. fps 1 missed short bottom positions; fps 3–4 merged a half-squat rep away; default resolution failed depth.
- After normalization every clip is `video/mp4` and 1–4 MB, so inline base64 is fine.
- The model may wrap JSON in ``` fences; strip them before `json.loads`.
- Timings: one Flash call 7–10 s; 10 Flash calls in parallel finish in 10–19 s. Argon is slower and variable in launch week (12 s typical, up to 60 s). Enforce timeouts.

## 4. Prompts (use verbatim)

PROMPT_FRONT:
```text
너는 맨몸 스쿼트 자세를 판정하는 코치야.
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
{"reps":[{"rep":1,"bottom":0.0,"upper":"곧음|말림","knee":"정렬|안쪽","depth":"판단 불가","pause":"충분|짧음|없음","low_confidence":[]}]}
```

PROMPT_SIDE: identical to PROMPT_FRONT except:
- "정면에서" → "측면에서"
- section 2 becomes: `2. 무릎 (knee)` / `- 측면 영상에서는 판단할 수 없으므로 항상 "판단 불가"로 써`
- section 3 becomes: `3. 깊이 (depth)` / `- 깊음: 고관절이 무릎 윗부분과 같은 높이이거나 더 낮음 (허벅지가 바닥과 평행하거나 그 이하)` / `- 얕음: 고관절이 무릎보다 확실히 높음`
- low_confidence items: `upper, depth, pause`
- JSON line: `{"reps":[{"rep":1,"bottom":0.0,"upper":"곧음|말림","knee":"판단 불가","depth":"깊음|얕음","pause":"충분|짧음|없음","low_confidence":[]}]}`

RECHECK (build from the axes that are disputed for that rep; `{view}` is 정면 or 측면):
```text
첨부한 영상 구간에는 {view}에서 찍은 맨몸 스쿼트 반복이 한 번 있어. 가장 낮은 지점을 기준으로 판정해.
{one line per disputed axis, from the list below}
바르게 한 반복은 바르다고 판정해. 문제를 억지로 찾지 마.
JSON만 출력해: {"upper": "...", "knee": "..."}   ← only the disputed axes
```
Axis lines:
- `- upper: 곧음(시선이 정면 또는 2~3m 앞 바닥, 가슴이 펴짐) / 말림(고개를 숙여 발 바로 앞을 보거나 등 위쪽이 둥글게 말림)`
- `- knee: 정렬(무릎이 두 번째 발가락 방향이거나 그보다 바깥) / 안쪽(두 무릎 사이 간격이 좁아지거나 무릎이 엄지발가락 쪽이나 그 안쪽으로 모임)`
- `- depth: 깊음(고관절이 무릎 윗부분과 같은 높이이거나 더 낮음) / 얕음(고관절이 무릎보다 확실히 높음)`
- `- pause: 충분(가장 낮은 지점에서 1초 이상 멈춤) / 짧음(멈췄지만 1초 미만) / 없음(바로 올라옴)`

COMPARE:
```text
첫 번째 영상은 지난 세션, 두 번째 영상은 오늘 세션이야. 둘 다 정면에서 찍은 같은 사람의 맨몸 스쿼트야.
상체(시선·등), 무릎 방향, 바닥 정지 세 가지에 대해 지난 세션 대비 오늘 무엇이 좋아졌고 무엇이 남았는지 말해줘.
근거가 되는 장면의 시각(영상 번호와 초)을 함께 적어. 각 항목 한두 문장, 영어로.
```

## 5. Decision logic (pure functions, unit-testable)

Value normalization (Korean → code). **Read only the text before any "(" and strip spaces** — Argon sometimes echoes the rule text, e.g. `없음(바로 올라옴)`.

| Axis | good | problem |
|---|---|---|
| upper | 곧음 → `S` | 말림 → `R` |
| knee (front only) | 정렬 → `A` | 안쪽 → `V` |
| depth (side only) | 깊음 → `D` | 얕음 → `H` |
| tempo (from pause) | 충분 → `C` | 짧음, 없음 → `B` |

"판단 불가" means the axis is not judged in that view; ignore it.

Rep alignment across the 5 Flash runs: take the most common rep count; use the first run with that count as the anchor; match each other run's reps to the anchor by nearest `bottom` (within 0.7 s, else "missing").

Per cell (rep × axis), with 5 votes:
1. All votes equal → **Confirmed**.
2. Otherwise → mark the rep for verification. One Argon RECHECK call per rep (covering all its disputed axes), 25 s timeout, all reps in parallel. If Argon answers → **Verified by Argon** with Argon's value. If timeout/error → **Please check this scene** with the majority value (tie → the problem value).

Human in the loop: every cell whose final value is a problem letter, and every "Please check this scene" cell, gets a "Check this scene" button that seeks the video to `bottom - 1.0` s and asks **Looks right / Not quite**. "Not quite" flips the cell to the good value. The type card shows "provisional" until every such cell has an answer.

Type (4 letters, order upper-knee-depth-tempo):
- For each axis and each view that judges it, the view's letter is the problem letter if more than half of that view's reps have the problem value.
- upper and tempo are judged in both views: problem if either view says problem. knee comes from the front view, depth from the side view.
- Look up the name and one-liner in `content.json → types`.

Recommendations (`content.json → recommendations`):
- Survey "pain" = Yes → show `messages.pain` and no exercises.
- Level column from survey "experience". If experience is Regular and the type has 3 or more problem letters, use Sometimes instead.
- Problem letters in priority V, R, H, B → take up to 2 rows. If none → the "all_clear" row.

## 6. UI (Streamlit, English, from content.json)

**DESIGN.md is binding for every screen and string.** Read it before writing any UI code and run its review checklist before each commit.

1. Header + `messages.footer` (form feedback, not medical advice; videos never stored).
2. Survey: 5 questions (`survey`). If time is short, keep only pain and experience.
3. Upload: front and side clips (mp4/mov, ≤ 30 MB, `messages.upload_guide`). One "Analyze" button.
4. While analyzing: rotate `tips` every few seconds.
5. Show first-pass results as soon as Flash finishes (≈10 s): a rep × axis table per view. Disputed cells show "Verifying with Gemini 4 Argon…", then update in place as each recheck returns.
6. Flagged cells: "Check this scene" → seek the clip (`st.video(data, start_time=int(seconds))`, keep the seek target in `st.session_state`) → Looks right / Not quite.
7. Type card (name, 4 letters, one-liner), then recommendation cards (exercise + one-liner).
8. "Compare with my last session": a second upload for the previous front clip → COMPARE → show Argon's answer.
9. A small "How it works" diagram: Flash screens → Argon verifies → you confirm.

## 7. Verification loop for you (the coding agent)

Write `tests/eval.py`: run the full pipeline on the test videos in `$TEST_VIDEO_DIR` and compare to `answer_key.json`. Print a per-cell table and a summary (correct / false alarm / missed / still unclear) and the predicted type.

Pass criteria (from pre-event experiments with this exact pipeline):
- `front_mixed_720p.mp4` + `side_mixed_720p.mp4`: all scored cells correct (19), 0 false alarms.
- `ref_front_5reps.mp4` + `ref_side_5reps.mp4`: all 30 cells correct, type SADC.
- `front_before_720p.mp4` + `side_before_720p.mp4`: type RVDB.
- "either" in the key means not scored. A cell left as "Please check this scene" is acceptable if its value is correct.

Also unit-test the pure functions (normalization, alignment, voting, type, recommendations) without API calls.

## 8. Milestones (commit and deploy after each)

| # | Goal | Done when |
|---|---|---|
| M0 | Repo with `.gitignore`, `requirements.txt`, Dockerfile, secret wired | No key anywhere in the repo |
| M1 | Upload one clip → normalize → one Flash call → raw JSON on screen, deployed to Cloud Run | A raw phone clip opens on a phone and returns JSON |
| M2 | 5 parallel Flash runs per clip (both clips in parallel), alignment, voting, table | Mixed clips show the planted faults |
| M3 | Argon recheck with progressive update and 25 s timeout | Mixed front rep 2 knee becomes "Verified by Argon: Caving in" |
| M4 | Human-in-the-loop confirm, type card, recommendations, survey (pain + experience) | Before clips → The Roller Coaster |
| M5 | Session comparison, tips, footer, "How it works" | Before vs ref front clips produce a timestamped comparison |
| M6 | `tests/eval.py` passes; `--min-instances 1` redeploy; README in English | All pass criteria met |

If behind schedule, drop in this order: survey questions 3–5, tips, session comparison. Never drop the Argon recheck or the human confirmation; they are judging criteria.

## 9. Deploy

```bash
gcloud services enable compute.googleapis.com run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
read -s -p "API key: " KEY && echo && printf "%s" "$KEY" | gcloud secrets create gemini-key --data-file=- && unset KEY
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding gemini-key \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
gcloud run deploy squattype --source . --region asia-northeast3 --allow-unauthenticated \
  --memory 2Gi --cpu 2 --session-affinity --min-instances 0 --max-instances 2 \
  --set-secrets GEMINI_API_KEY=gemini-key:latest
```
Dockerfile CMD: `streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --server.maxUploadSize=30 --server.enableXsrfProtection=false` (XSRF off from the start: Streamlit uploads often return 403 behind Cloud Run).
`requirements.txt`: `streamlit`, `google-genai>=2.29`, `imageio-ffmpeg`.
Cloud Run rejects request bodies over 32 MB before the app sees them; that is why the limit is 30 MB and the guide asks for 1080p 30 fps recording.
