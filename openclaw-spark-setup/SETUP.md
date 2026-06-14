# DGX Spark setup for LLaMA-Factory (openclaw runbook)

Goal: prepare **each** NVIDIA DGX Spark so the StartOS "LLaMA-Factory" service can
SSH in and run fine-tuning jobs in Docker. Run every step **on the Spark** (over
SSH or locally). If there are two Sparks, do this on **both**.

This bundle contains exactly what the Spark needs:
- `Dockerfile` — the LLaMA-Factory image (aarch64 + CUDA 12.8 for GB10 `sm_121`)
- `build.sh` — builds that image
- `train_entrypoint.sh` — the in-container training entrypoint (baked into the image)

The Spark does **not** need the training data or configs — the StartOS service
rsyncs those in at run time and launches `docker run` itself.

---

## Variables (set these first)
```bash
SPARK_USER=nvidia                       # the SSH login user on this Spark
WORKDIR=/home/$SPARK_USER/llamafactory-runs   # must match the service's "Remote Work Directory"
IMAGE=llamafactory-spark:latest         # must match the service's "Spark Image Tag"
# Paste the StartOS service's SSH *public* key (the .pub half of the key you put
# into the "Configure Sparks" action). openclaw should obtain this from the user.
SERVICE_PUBKEY="ssh-ed25519 AAAA... llama-factory@startos"
```

## 1. Verify the GPU + Docker stack
DGX OS ships Docker + the NVIDIA Container Toolkit. Confirm GPU containers work:
```bash
nvidia-smi -L
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.04-py3 nvidia-smi -L
```
Both must list the GB10 GPU. If the second fails, the NVIDIA Container Toolkit
isn't wired up — fix that before continuing (`nvidia-ctk runtime configure
--runtime=docker && sudo systemctl restart docker`).

## 2. Grant the StartOS service SSH access
The service authenticates with a key; add its **public** half:
```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "$SERVICE_PUBKEY" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

## 3. Build the LLaMA-Factory image
```bash
# from this bundle's directory, on the Spark:
IMAGE="$IMAGE" bash build.sh
```
Notes:
- First build pulls NVIDIA's NGC PyTorch base (several GB) and compiles deps —
  expect 15–40 min.
- If the build fails on CUDA/arch, bump the base tag to a newer DGX-Spark-tuned
  NGC PyTorch image (must be **aarch64** with **CUDA ≥ 12.8**):
  `NGC_TAG=25.06-py3 IMAGE="$IMAGE" bash build.sh`.
- To avoid building twice, build on Spark #1 then copy to Spark #2:
  `docker save "$IMAGE" | ssh $SPARK_USER@<spark2> docker load`

## 4. Prepare the work directory
```bash
mkdir -p "$WORKDIR"      # the service also mkdir -p's this, but pre-create it
```

## 5. Verify
```bash
docker image inspect "$IMAGE" >/dev/null && echo "image OK"
docker run --rm --gpus all "$IMAGE" \
  python -c "import torch,llamafactory; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```
Expect `cuda True` and a torch version built for CUDA ≥ 12.8.

From the StartOS side, the **Test Spark Connection** action should now succeed.

---

## Two-node training (only if using both Sparks)
The 31B LoRA fits on one Spark; do this only for full fine-tuning / larger models.
- Ensure the two Sparks reach each other over the **ConnectX / 200GbE** link
  (not just the slow management NIC). `ping` the peer's ConnectX IP.
- The service launches each node with `--network host` and `MASTER_PORT=29500`.
  Make sure that port is open between the two Sparks (default DGX OS firewall
  allows LAN; verify if you've hardened it).
- `MASTER_ADDR` is the **primary** Spark's address as configured in the service.
  Confirm the secondary Spark can resolve/route to it on the ConnectX subnet.
