# FinAgent — Your Hackathon Submission Checklist

**You're driving from here. I'll help with any step — just tell me what you're on.**

---

## ✅ Already done (by me)

1. All four specs (orchestration, tools, inference, frontend) built and tested — 309 tests passing.
2. Cross-spec dependency conflicts resolved.
3. `gradio-frontend/space/` — a ready-to-push Hugging Face Space directory.
4. `DEPLOY_AMD.md` — step-by-step commands for the AMD Developer Cloud instance.
5. `DEPLOY_HF.md` — step-by-step commands for pushing to Hugging Face Spaces.
6. `DEPLOYMENT_PLAN.md` — the overall plan.

---

## 🎯 Your step-by-step (rough time: 4–5 hours)

### Step 1 — Push your code to GitHub (20 min)

The hackathon requires a public GitHub repo. If yours is private or not yet pushed, let's fix that:

```bash
cd /Users/emmanuelakbi/Documents/Projects/React-JS-TS/FinAgent

# Check if you have a remote
git remote -v

# If there's no remote, create a new public repo on github.com first
# Then:
git remote add origin https://github.com/YOUR-USERNAME/FinAgent.git
git branch -M main
git add .
git commit -m "FinAgent: multi-agent trading signals on AMD MI300X"
git push -u origin main
```

If git complains about existing commits, tell me the error — we'll sort it.

### Step 2 — Deploy vLLM on AMD Cloud (~1 hr)

Open `DEPLOY_AMD.md`. Follow steps 1–7 in order.

When you see vLLM print `Application startup complete` and the `/v1/models` endpoint returns JSON — **stop and tell me the URL**. I'll verify it works.

### Step 3 — Deploy the Hugging Face Space (~30 min)

Open `DEPLOY_HF.md`. Follow steps 1–4.

When the Space shows a signal card for AAPL — **stop and tell me**. Send me the Space URL.

### Step 4 — Record the demo video (~1 hr)

Outline I'll write for you — just narrate and screen-record. Target: 2–3 minutes.

**Script (I'll draft after Step 3 is working):**

1. (0:00–0:15) Problem: traders want to scan many tickers fast; existing AI tools are shallow one-shot prompts.
2. (0:15–0:45) Solution: 5 specialized agents collaborate. Screen-record the Space as agents run.
3. (0:45–1:30) Walk through a signal card: BUY with confidence 78%, entry/stop/target, per-agent reasoning.
4. (1:30–2:15) Architecture: show the HF Space ↔ AMD MI300X diagram. Mention Qwen3-8B via vLLM on ROCm.
5. (2:15–3:00) Closing: GitHub link, try it yourself, tag AMD and lablab.

Record with QuickTime (Mac) or OBS. Upload to YouTube as unlisted, paste the link into the lablab submission form.

### Step 5 — Slide deck (~30 min)

I'll write a 6-slide Google Slides outline. You paste it in and style it. Topics: problem, solution, architecture, demo screenshot, tech stack, team.

### Step 6 — Cover image (~15 min)

Use <https://postermywall.com> or Canva, or grab a screenshot of your Space with a clean ticker + BUY signal card. 1200×675 works.

### Step 7 — Submit on lablab (~20 min)

Go to <https://lablab.ai/ai-hackathons/amd-developer> → Enroll → "Submit Your Project". Fill:

- **Title:** FinAgent: Multi-Agent Trading Signals on AMD MI300X
- **Short description:** "Five specialized AI agents collaborate on AMD MI300X (Qwen3-8B + vLLM) to produce structured BUY/SELL/HOLD trading signals with confidence, entry/stop/target prices, and per-agent reasoning."
- **Long description:** copy your HF Space README (adapted)
- **Category tags:** AI Agents, AMD Developer Cloud, AMD ROCm, CrewAI, Qwen3
- **Cover image:** upload
- **Video:** YouTube link
- **Slides:** Google Slides link (or PDF)
- **GitHub repo:** your FinAgent URL
- **Demo App Platform:** Hugging Face Spaces
- **Application URL:** the Space URL

Click Submit.

### Step 8 — Build in Public (the extra prize, optional but easy) (~15 min)

Tweet / LinkedIn post. I'll give you the wording:

> Just shipped FinAgent for the @AMD Developer Hackathon 🚀
>
> 5 specialized AI agents collaborate on a single AMD Instinct MI300X running Qwen3-8B + vLLM to produce structured trading signals.
>
> Full pipeline in <60s per ticker. Live: [HF Space link]
> Open source: [GitHub link]
>
> @lablabai @AIatAMD

Post one before submission, one after. Qualifies you for the Build-in-Public prize pool.

---

## 🚨 If I don't hear from you

If you get stuck, send me **any one of these**:

- An error message
- A screenshot
- "I don't see the button you mentioned"
- "What do I click next"

Anything. I'll unblock you.

---

## 🏁 Deadline recap

**May 10, 2026, 8:00 PM WAT.** That's ~24 hours from the start of this session.

Priority order if time runs short:

1. ✅ Must-have: GitHub repo + HF Space (live) + filled submission form
2. Should-have: video
3. Nice-to-have: slides, build-in-public posts

The Space + GitHub are the minimum to qualify. Everything else is bonus polish.
