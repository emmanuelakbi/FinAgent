# FinAgent Deployment Plan — Hackathon Submission

**Deadline:** May 10, 2026, 8:00 PM WAT (~24 hours from now)
**Track:** AI Agents & Agentic Workflows
**Goal:** Live Hugging Face Space backed by vLLM on AMD MI300X, showing a 5-agent CrewAI pipeline that generates trading signals.

---

## Success criteria (what judges see)

1. **Live HF Space** at `https://huggingface.co/spaces/<you>/finagent` — clickable, no login needed, analyzes any ticker the judge types in.
2. **GitHub repo** — the source code, cleanly organised.
3. **Cover image + video + slides** — 2–3 min walkthrough.
4. **AMD MI300X actually in use** — the Space's backend traffic hits a vLLM server running on the AMD Developer Cloud. Visible in activity feed, screenshots, or the architecture diagram.
5. **Qwen used meaningfully** — qualifies for the Qwen Special Reward.

---

## Phased timeline (checkpoint-based)

### Phase 1 — Frontend hardening (✅ done by me, 0 hr)

- Clean up gradio/websockets dependency conflict.
- Package `gradio-frontend/` so it drops into a HF Space with a single `app.py + requirements.txt + crew/ + tools/`.

### Phase 2 — AMD Developer Cloud vLLM deployment (~2 hr, mostly waiting)

- You log into AMD Developer Cloud web console.
- I guide you step-by-step to provision an MI300X instance.
- You SSH in (or use their web terminal), paste 4–5 commands.
- vLLM boots, serves `Qwen/Qwen3-8B` on port 8000.
- We expose it via a public URL (or use a tunnel if the cloud doesn't give us one).

### Phase 3 — Hugging Face Space deployment (~30 min)

- You create a new Space under your HF account (Gradio SDK).
- `git clone` → copy deployment package → `git push`.
- Set `VLLM_ENDPOINT_URL` as a Space secret.
- Space builds and launches. We click it, run one analysis, confirm it works.

### Phase 4 — Submission materials (~2 hr)

- README polish.
- Cover image (I'll provide ideas).
- Video script (I'll write the narration, you record screen).
- Slide deck (I'll draft content, you paste into Google Slides).

### Phase 5 — Final check + submit (~30 min)

- Run judge-like test: open the Space in incognito, type AAPL, MSFT, confirm output.
- Fill the lablab submission form.
- Tweet-style "build in public" posts (qualifies for the extra challenge).

---

## Estimated cost burn on AMD credits

- MI300X on AMD Developer Cloud: roughly $2–5/hour.
- We need: ~2 hr to deploy + ~1 hr for video recording + buffer for judge review window.
- Budget: $20–30 of your $100 credit. Plenty of headroom.
- **Strategy:** only keep the instance running during development, video recording, and the active judging window. Shut it down when idle.

---

## What I'm doing next (autonomous)

1. Produce a deployable Space package in `gradio-frontend/space/` containing:
   - Copy of `app.py`, `validation.py`, `rendering.py`
   - Copy of the `crew/` package (needed at runtime)
   - `requirements.txt` with HF-compatible pins
   - Space-compatible `README.md` header
2. Write a `DEPLOY_AMD.md` with step-by-step commands you'll run on AMD Developer Cloud.
3. Write a `DEPLOY_HF.md` with step-by-step commands for Hugging Face.

After I finish these, we stop and you drive the deployment with my guidance.
