# LLaMA-Factory (DGX Spark Orchestrator)

This service is a **control plane**. It does not train on your StartOS server.
It dispatches LLaMA-Factory fine-tuning jobs over SSH to one or two **NVIDIA DGX
Sparks**, streams the logs back, and retrieves the trained adapter.

## One-time setup

### 1. Prepare each Spark
LLaMA-Factory ships no ARM64/Blackwell image, so each Spark needs the
`llamafactory-spark:latest` image built on it, plus SSH access for this service.
Follow the separate **Spark setup bundle** (`openclaw-spark-setup/SETUP.md`),
which builds the image, authorizes the service's SSH key, and verifies the GPU.

### 2. Create an SSH key for this service
On any machine:
```bash
ssh-keygen -t ed25519 -f ./id_spark -N ''
ssh-copy-id -i ./id_spark.pub nvidia@<spark-host>   # repeat for the 2nd Spark
```

### 3. Configure the service (StartOS Actions)
- **Configure Sparks** — host(s), SSH user/port, paste the **private** `id_spark`,
  and (optionally) a Hugging Face token. Gemma 4 is Apache-2.0 so a token is
  usually unnecessary.
- **Training Defaults** — base model (default `google/gemma-4-31B-it`),
  LoRA/hyperparameters, bf16 vs 4-bit.
- **Test Spark Connection** — verifies SSH + GPU + that the image is built.

## Running a fine-tune
Open the **Web UI**:
1. Upload your training data — a `.jsonl` where each line is
   `{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`.
2. Pick the dataset, optionally override per-run hyperparameters, **Start fine-tune**.
3. Watch live logs. When done, **Fetch adapter ↓** copies the LoRA adapter back
   into this service's volume (under `runs/<id>/output`), included in backups.

## Notes
- A 31B LoRA fits on a **single** Spark (128 GB unified memory). Enable
  "Use Both Sparks" only for full fine-tuning or larger models — it runs 2-node
  DeepSpeed ZeRO-3 over your ConnectX link and needs more tuning.
- With a very small dataset, keep epochs low (1–3) to avoid overfitting.
- This service holds an SSH key that can run containers on your Sparks. Treat it
  as privileged; use a dedicated key.
