# Deploying vLLM on AMD Developer Cloud (MI300X)

**Goal:** Get a public URL serving Qwen3-8B via vLLM on an MI300X instance, that the Hugging Face Space can hit.

**Prereqs:**

- AMD Developer Cloud account with $100 credit (✅ you have this)
- A web browser
- ~30 minutes of attention; ~2 hours of machine time total

---

## Step 1 — Provision the instance (5 min)

1. Open <https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html> and sign in with your AMD Developer account.
2. Click **Launch Developer Cloud** (the button name may vary — it opens the console).
3. From the console dashboard, click **Create Instance** (or **Deploy**).
4. Select:
   - **GPU:** AMD Instinct MI300X (192 GB HBM)
   - **Image:** whichever pre-built ROCm 6.2 image they offer. If you see choices like "ROCm 6.2 + PyTorch 2.4" or "Ubuntu 22.04 + ROCm" — pick the one that mentions **ROCm 6.2**. If there's only one option, pick that.
   - **Instance size:** 1× MI300X (single GPU is enough for Qwen3-8B).
   - **Networking:** **Public IP required**. Make sure there's a public IP or equivalent option enabled. We need to reach port 8000 from the internet.
5. Click **Launch** / **Deploy** / whatever the green button says.
6. Wait ~2–3 minutes for the instance to boot. You should see a public IP address or hostname once it's ready. Write it down. Example: `203.0.113.45` or `mi300x-xyz.amd.cloud`.

**Cost:** ~$2–5/hour while running. The $100 credit gives you 20–50 hours. Shut it down when not in use.

---

## Step 2 — SSH into the instance (2 min)

The AMD Developer Cloud console will usually give you one of these options:

- A browser-based web terminal (easiest for non-technical use — just click "Open Terminal")
- Or an SSH command like `ssh user@203.0.113.45` with a key you downloaded

**Use whichever works.** The commands below are the same either way.

Once you're in, you should see a Linux prompt like `user@mi300x:~$`.

Quick sanity check — paste this:

```bash
rocm-smi
```

You should see a table listing one or more GPUs with "MI300X" in the name. If you see "command not found" or "no GPUs detected," tell me and we'll troubleshoot.

---

## Step 3 — Clone the FinAgent inference setup (1 min)

Paste this into the terminal:

```bash
git clone https://github.com/YOUR-USERNAME/FinAgent.git
cd FinAgent/inference
chmod +x setup.sh health_check.sh
```

Replace `YOUR-USERNAME` with your actual GitHub username. If your repo is private, you'll need to either:

- Make it public first (safest for this hackathon — open source is required anyway per the "Build in Public" challenge)
- Or use a personal access token when cloning

---

## Step 4 — Launch the vLLM server (10–20 min on first run)

Paste this:

```bash
./setup.sh --host 0.0.0.0 --port 8000
```

What happens:

1. The script detects your GPU and installs vLLM + dependencies if not already present (~5–10 min first run only).
2. It downloads the Qwen3-8B model from Hugging Face (~15 GB, ~2–5 min).
3. It launches vLLM serving the model on port 8000.

When you see a line like:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

the server is **live**. Leave this terminal open.

---

## Step 5 — Verify it works (1 min, new terminal or new tab)

Open a second terminal / tab on the same machine and paste:

```bash
cd FinAgent/inference
./health_check.sh --host 0.0.0.0 --port 8000
```

Expect: `HEALTHY: <some Qwen response>` and exit code 0.

If it says `UNHEALTHY`, copy-paste the output and I'll help debug.

---

## Step 6 — Make it reachable from the internet

Two options:

### Option A — Direct public IP (easiest, if the instance has one)

Check your instance's public IP in the AMD console. From your laptop, open a browser and visit:

```
http://<public-ip>:8000/v1/models
```

You should see a JSON list mentioning `Qwen/Qwen3-8B`. If you do, your endpoint is:

```
http://<public-ip>:8000/v1
```

Write this down. This is what goes into the HF Space secret.

### Option B — Cloudflare Tunnel (if direct IP doesn't work)

If port 8000 is firewalled or the instance only has a private IP, we'll use a free Cloudflare tunnel. Tell me and I'll send the 3-command setup — it creates a public `https://...trycloudflare.com` URL that forwards to your vLLM server.

---

## Step 7 — Keep it running during the judging window

**Critical:** the instance must stay up during the demo video recording AND the judging window. If it shuts down, the Space falls back to the configuration-error banner.

Set a calendar reminder to check it every few hours.

**To shut down** (after submission is judged or to save credits):

In the AMD console, click **Stop** or **Delete** on the instance. You pay nothing while it's stopped.

---

## Troubleshooting

| Symptom                                                     | Fix                                                                          |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `rocm-smi: command not found`                               | Wrong image selected. Re-provision with ROCm 6.2 image.                      |
| Model download hangs                                        | Usually flaky HF CDN. Retry: `./setup.sh` again.                             |
| `out of memory` during model load                           | Try: `./setup.sh --model Qwen/Qwen3-4B` (smaller model, still high quality). |
| `port 8000 in use`                                          | A previous vLLM is still running. Run `pkill -f vllm` then retry.            |
| Space says "Configuration error: VLLM_ENDPOINT_URL not set" | Secret not saved in HF. See DEPLOY_HF.md step 3.                             |
| Space says "Connection refused"                             | vLLM isn't reachable. Check the URL works from your laptop's browser.        |

---

When this is all done, tell me **"vLLM is live at http://..."** and we'll move on to the Hugging Face Space deployment.
