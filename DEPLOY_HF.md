# Deploying FinAgent to Hugging Face Spaces

**Goal:** Publish the Gradio frontend as a public HF Space that anyone (including judges) can open in a browser.

**Prereqs:**

- HF account with the hackathon organization joined (<https://huggingface.co/organizations/lablab-ai-amd-developer-hackathon/share/ELARrxoRIHvseSHRhANJYFEZQazsQIYhJf>)
- vLLM running on AMD Cloud (see `DEPLOY_AMD.md`)
- ~15 minutes

---

## Step 1 — Create the Space (3 min)

1. Go to <https://huggingface.co/new-space>
2. Fill in:
   - **Owner:** `lablab-ai-amd-developer-hackathon` (the hackathon org — joining it earns you eligibility for the HF Special Prize)
   - **Space name:** `finagent` (or something unique you like — check it's available)
   - **License:** MIT
   - **Space SDK:** **Gradio**
   - **Space hardware:** **CPU basic (free)** — the heavy lifting happens on your AMD instance, not here.
   - **Visibility:** **Public**
3. Click **Create Space**.

HF shows you a repo page with a "Clone this Space" section.

---

## Step 2 — Push the code (5 min)

Copy the URL HF gave you (looks like `https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/finagent`).

On your laptop, in the FinAgent project directory, run:

```bash
# Clone the empty Space repo into a temp working dir
cd /tmp
git clone https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/finagent
cd finagent

# Copy the prepared deployment package in
cp -r /Users/emmanuelakbi/Documents/Projects/React-JS-TS/FinAgent/gradio-frontend/space/* .

# Commit and push
git add .
git commit -m "Initial FinAgent deployment"
git push
```

HF may prompt for your username/token. Use:

- Username: your HF handle
- Password: an **access token** (generate at <https://huggingface.co/settings/tokens>, Write scope)

After `git push` succeeds, go back to the Space page in your browser. HF starts building the container automatically. Watch the **Logs** tab — it takes 3–5 minutes.

---

## Step 3 — Set the VLLM_ENDPOINT_URL secret (2 min)

On your Space page:

1. Click **⚙️ Settings** at the top of the Space.
2. Scroll to **Repository secrets**.
3. Click **New secret**.
4. Name: `VLLM_ENDPOINT_URL`
5. Value: the URL from `DEPLOY_AMD.md` step 6 — something like `http://203.0.113.45:8000/v1`
6. Click **Save**.
7. Go back to the **App** tab. The Space will automatically restart; wait ~30 seconds.

---

## Step 4 — Test it (2 min)

1. Open the Space in your browser (the public URL is `https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/finagent`).
2. In the "Watchlist" box, type: `AAPL`
3. Click **🔍 Analyze**.

Expect:

- The Analyze button goes dim.
- The activity feed shows lines like "System: Analysis started for 1 ticker(s)" then "Market Scanner started task for AAPL" etc.
- After 30–90 seconds, a signal card appears with action (BUY/SELL/HOLD), confidence, prices, and reasoning.

If you see `❌ Configuration error: VLLM_ENDPOINT_URL environment variable is not set.`:

- Go back to Step 3, make sure the secret name is spelled exactly right (case-sensitive).
- Check the Space logs for any import errors.

If you see `❌ An error occurred: ...`:

- The error message will usually tell you: network, auth, or model issue.
- Most likely the vLLM endpoint isn't reachable — open the URL in a new tab and see if `<your-endpoint>/v1/models` returns JSON.

---

## Step 5 — Claim the HF Special Prize

Once the Space is working:

1. Share it on X/LinkedIn with `@lablabai` and `@AIatAMD` tagged.
2. Ask a few friends to click the ❤️ Like button on the Space — the prize goes to the Space with the most likes at the deadline.
3. Post a screenshot of a signal card with an interesting ticker.

---

## Troubleshooting

| Symptom                                     | Fix                                                                                                                                                                  |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Build fails with `ModuleNotFoundError`      | A package in `requirements.txt` didn't install. Check HF build logs for the real error.                                                                              |
| Build fails with lancedb / tokenizers error | Usually transient. Click "Factory rebuild" in Settings.                                                                                                              |
| Space UI loads but Analyze hangs forever    | vLLM endpoint unreachable. Verify it via `curl <url>/v1/models` from your laptop.                                                                                    |
| Signal card shows "Failed to parse"         | vLLM returned output in an unexpected format. Usually means the model didn't follow the prompt — retry, and if persistent, switch to Qwen3-14B if your GPU has room. |
| Space is in "Runtime error" state           | Click the **Logs** tab — the Python traceback will be there. Copy it to me.                                                                                          |

---

## If something goes wrong and you're stuck

Tell me:

1. What step you're on
2. What error you see (screenshot or copy-paste)

I'll fix it or give you the next command.
