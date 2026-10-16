# DESIGN — SquatType

Follow this for every screen and every string. The user's own video is the visual centre of the app; everything else stays quiet around it. When in doubt, remove.

## 1. Voice

- A calm coach who has watched your clip. Specific, short, never hyped.
- Witty lines live in exactly three places: type one-liners, exercise one-liners, loading tips (all in `content.json`). Everywhere else is plain and factual.
- Sentence case for every title, label and button ("Analyze my squat", not "Analyze My Squat").
- Say what happened with numbers and rep indexes: "Knees caved in on reps 2 and 4." Not "Some issues were detected."
- Never: emoji, exclamation marks, "AI-powered", "revolutionary", "unlock", "journey", "seamless", "Welcome to…", filler sentences, made-up statistics.
- Never medical words (diagnosis, injury risk, rehab, treatment). This is form feedback.

Glossary (use these words only): clip, rep, session, type, scene, "Caving in", "Verified by Argon", "Please check this scene".

## 2. Theme

`.streamlit/config.toml`:
```toml
[theme]
base = "light"
primaryColor = "#0F766E"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F4F4F2"
textColor = "#1F2328"
```

| Role | Colour | Use |
|---|---|---|
| Accent | `#0F766E` (teal) | primary button, links, the type letters. Nothing else. |
| Good | `#15803D` | good values (Straight, Aligned, Deep, Controlled) |
| Problem | `#B45309` (amber) | problem values. Never red: this is feedback, not an alarm. |
| Unclear | `#475569` (slate) | "Please check this scene", "Verifying…" |
| Surface | `#F4F4F2` | cards and table background |

- One accent colour only. No gradients, shadows, glass effects, or background images.
- Colour is never the only signal: every coloured value also carries its word ("Caving in", not just an amber dot).
- Keep custom CSS to the minimum needed for the coloured value labels. Prefer native Streamlit components.

## 3. Layout

- `st.set_page_config(page_title="SquatType", layout="centered")`. One column. It must work at phone width (≈390 px); check it on a phone.
- One `st.title` (the app name), then `st.subheader` per step. No other heading levels.
- Numbered steps, in this order: 1 About you → 2 Your clips → 3 Results → 4 Compare sessions.
- Spacing comes from Streamlit's defaults and `st.container(border=True)`. Do not stack bordered containers inside bordered containers.
- No `st.balloons`, `st.snow`, confetti, mascots, generated illustrations, stock photos or icon sets.

## 4. Screens

Upload (step 2):
```
2  Your clips
   Camera at hip height, 2–3 m away, head to toe in frame…   (messages.upload_guide, small grey text)
   [ Front clip  (mp4/mov) ]   [ Side clip  (mp4/mov) ]       (two columns on desktop, stacked on phone)
   [ Analyze my squat ]                                       (primary button, full width)
```

While analyzing: one `st.status` box with real step names and live counts, plus one rotating tip below it.
```
▸ Preparing your clips…                                   done in 5 s
▸ Screening both clips with Gemini 3.8 Flash (10 runs)…  done in 11 s
▸ Verifying 2 reps with Gemini 4 Argon…                   rep 2 done, rep 4 running
  Tip: Pick a spot 2–3 m ahead on the floor. That's your eyes' reserved seat.
```
No fake progress bars and no percentages that do not come from real work.

Results (step 3), top to bottom:
```
┌───────────────────────────────────────────────┐
│ The Roller Coaster                            │  name, large
│ R  V  D  B                                    │  4 letters, accent colour, monospace
│ Upper body · Knees · Depth · Tempo            │  axis labels under the letters, small grey
│ All the drops, none of the brakes.            │  one-liner
│ Provisional until you check 3 scenes.         │  only while confirmations are pending
└───────────────────────────────────────────────┘

Front clip                          Side clip            (tabs or two sections)
Rep │ Upper body │ Knees               │ Tempo
 1  │ Straight   │ Aligned             │ Controlled
 2  │ Straight   │ Caving in · Verified│ Bouncy        [Check this scene]
 …
                                       ← each problem / unclear cell has its own small "Check this scene" button
[ video player, seeks to the chosen scene ]
Is this right?  [ Looks right ]  [ Not quite ]

Try this next  (General exercise examples)
┌ Band squat ─────────────────────────────────┐
│ The band pulls your knees in. Don't let it win. │
│ Why: knees caved in on reps 2 and 4.            │
└─────────────────────────────────────────────────┘
```
- Status words next to values: "Verified by Argon" in slate small text; "Confirmed" is not shown (it is the default).
- The "Why" line is generated in code from the rep indexes, not by a model.

Compare (step 4): one upload for the previous front clip, one button "Compare with my last session", then Argon's answer as plain text with its timestamps. No charts.

How it works (footer area, collapsed `st.expander`): three short lines, no diagram library.
```
1  Gemini 3.8 Flash screens every rep five times.
2  Gemini 4 Argon re-checks only the reps where the votes split.
3  You confirm anything still unclear.
```
Then `messages.footer` in small grey text.

## 5. States

| State | What the user sees |
|---|---|
| No clips yet | Button disabled; the upload guide is the only hint. |
| Argon timeout on a rep | That cell shows "Please check this scene" and the reason in one line: "Argon took too long on rep 3." |
| Model returns bad JSON | Retry once silently; if it fails again, "Couldn't read the result for the side clip. Try again." with a retry button. |
| Pain = Yes | Results show as usual; the recommendation section shows `messages.pain` instead of cards. |
| Clip over 30 s or 30 MB | `messages.too_long` before any API call. Clips of 20–30 s are accepted. |

## 6. Review before every commit

Scan the diff and the running app for:
- [ ] any emoji or exclamation mark in UI strings
- [ ] any colour outside the table in section 2, any gradient or shadow
- [ ] any string not from `content.json` that sounds like marketing
- [ ] a value shown by colour alone
- [ ] layout broken at phone width
- [ ] words outside the glossary for the same thing (e.g. "video" vs "clip" mixed on one screen)
