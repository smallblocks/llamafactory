# LLaMA-Factory — StartOS service

A StartOS (Start9) package that turns **LLaMA-Factory** into a one-click fine-tuning
control panel for your **NVIDIA DGX Sparks**.

It does **not** train on the StartOS box. StartOS containers don't get GPUs, and
the Sparks already run the optimized NVIDIA stack — so this service is an
**orchestrator**: a web UI + SSH client that ships a job to the Spark(s), runs
LLaMA-Factory there in Docker, streams the logs back, and rsyncs the adapter home.

```
┌──────────────┐   SSH / rsync    ┌──────────────────────────────┐
│  StartOS box │ ───────────────▶ │  DGX Spark (GB10, 128GB, ARM) │
│  (this s9pk) │   docker run     │  llamafactory-spark image     │
│  web UI      │ ◀─────────────── │  GPU LoRA train → adapter     │
└──────────────┘   logs / adapter └──────────────────────────────┘
                                   (optionally a 2nd Spark, 2-node)
```

> **Spark prep is a separate bundle.** The image + SSH setup each Spark needs
> lives in `openclaw-spark-setup/` (delivered alongside this repo), not here.

## How GitHub builds the `.s9pk`
Push to `master`/`main` (or open a PR) → the **Build** workflow runs the Start9
shared pipeline, which installs `start-cli` + Node + Docker buildx/QEMU, runs
`make` (bundles `startos/` with `ncc`, then `start-cli s9pk pack` for x86_64 and
aarch64), and uploads each `*.s9pk` as a downloadable **artifact** on the run.

Tag `v0.1.0` → the **Release** workflow attaches the `.s9pk`s to a GitHub Release.
The release workflow signs packages, so set a repo secret **`DEV_KEY`**:
```bash
start-cli init-key                 # writes ~/.startos/developer.key.pem
# paste that PEM as the GitHub Actions secret named DEV_KEY
```
(The Build workflow needs no secret — it auto-generates an ephemeral key.)

## Build locally (optional)
Requires Docker + `start-cli` (Linux):
```bash
make            # -> llama-factory_x86_64.s9pk, llama-factory_aarch64.s9pk
make install    # install to the StartOS host in ~/.startos/config.yaml
```

## Layout
| Path | What |
|------|------|
| `startos/` | StartOS service code (TypeScript, `@start9labs/start-sdk` 1.5.3): `manifest/`, `main`, `interfaces`, `actions/`, `file-models/`, `versions/`. Bundled by `ncc` to `javascript/index.js`. |
| `orchestrator/` | Control-plane app baked into the service image: FastAPI (`app.py`), SSH/rsync helper (`spark_client.py`), LLaMA-Factory config generator (`lf_config.py`), web UI. |
| `orchestrator.Dockerfile` | The service image (root build context). |
| `configs/` | Reference `dataset_info.json` + `gemma4_31b_lora.yaml`. |
| `icon.svg`, `instructions.md`, `LICENSE` | Marketplace assets (root, per Start9 convention). |
| `Makefile`, `s9pk.mk` | Standard Start9 build (`s9pk.mk` is vendored plumbing — don't edit). |
| `.github/workflows/` | `build.yml` (artifacts) and `release.yml` (tagged releases). |

## Using it (after install)
See [`instructions.md`](instructions.md). Short version: prep each Spark via the
`openclaw-spark-setup/` bundle → run the **Configure Sparks** / **Training
Defaults** / **Test Spark Connection** actions → open the **Web UI**, upload your
`.jsonl`, **Start fine-tune**, watch logs, **Fetch adapter**.

## Training data format
One JSON object per line, OpenAI `messages` style:
```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## Known integration risks
- **No official LLaMA-Factory ARM64/Blackwell image** — built on the Spark from
  the `openclaw-spark-setup/` bundle (NGC PyTorch base, CUDA ≥ 12.8 for `sm_121`).
- **Gemma 4 chat template** defaults to `gemma`; switch to `gemma4` in the
  *Training Defaults* action if your LLaMA-Factory build ships it.
- **2-node training** (ConnectX DeepSpeed ZeRO-3) is wired but experimental;
  single-Spark LoRA is the robust default and fits a 31B.

## License
Apache-2.0 (matching LLaMA-Factory and Gemma 4).
