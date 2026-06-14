"""LLaMA-Factory orchestrator: a small control plane that dispatches fine-tune
jobs to remote DGX Sparks over SSH and streams logs back.

Job model: each run is a `docker` container on the Spark named `lf-<run_id>`.
Docker is the remote job manager — start detached, follow with `docker logs -f`,
check state with `docker inspect`, stop with `docker rm -f`. The run dir
(config + dataset) is rsynced up; the output adapter is rsynced back.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import time

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import lf_config
import spark_client as sc

DATA_DIR = os.environ.get("LF_DATA_DIR", "/data")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
RUNS_DIR = os.path.join(DATA_DIR, "runs")
HF_TOKEN_PATH = os.path.join(DATA_DIR, "secrets", "hf_token")

app = FastAPI(title="LLaMA-Factory Orchestrator")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

for d in (DATASETS_DIR, RUNS_DIR):
    os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------------- helpers
def _cfg() -> dict:
    try:
        return sc.load_config()
    except FileNotFoundError:
        raise HTTPException(400, "Not configured yet. Run the 'Configure Sparks' action.")


def _safe_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _run_path(run_id: str) -> str:
    if not re.fullmatch(r"[0-9A-Za-z._-]+", run_id):
        raise HTTPException(400, "bad run id")
    return os.path.join(RUNS_DIR, run_id)


def _hf_token() -> str | None:
    if os.path.exists(HF_TOKEN_PATH):
        t = open(HF_TOKEN_PATH).read().strip()
        return t or None
    return None


def _remote_run_dir(cfg: dict, run_id: str) -> str:
    return os.path.join(cfg.get("remoteWorkDir", "/home/ten31spark/llamafactory-runs"), run_id)


def _write_run_meta(run_id: str, meta: dict) -> None:
    with open(os.path.join(_run_path(run_id), "run.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _read_run_meta(run_id: str) -> dict:
    p = os.path.join(_run_path(run_id), "run.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# ----------------------------------------------------------------------------- UI
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/config")
def get_config():
    cfg = _cfg()
    cfg.pop("hfTokenSet", None)
    return cfg


@app.get("/api/datasets")
def list_datasets():
    files = sorted(f for f in os.listdir(DATASETS_DIR) if f.endswith(".jsonl"))
    out = []
    for f in files:
        path = os.path.join(DATASETS_DIR, f)
        n = sum(1 for _ in open(path, "rb"))
        out.append({"name": f, "examples": n, "bytes": os.path.getsize(path)})
    return out


@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...)):
    name = os.path.basename(file.filename or "")
    if not name.endswith(".jsonl"):
        raise HTTPException(400, "Dataset must be a .jsonl file")
    dest = os.path.join(DATASETS_DIR, name)
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"ok": True, "name": name}


# ----------------------------------------------------------------------------- runs
@app.post("/api/train")
async def start_train(request: Request):
    body = await request.json()
    cfg = _cfg()
    cfg.update({k: v for k, v in (body.get("overrides") or {}).items() if v not in (None, "")})

    dataset = os.path.basename(body.get("dataset", ""))
    src = os.path.join(DATASETS_DIR, dataset)
    if not dataset or not os.path.exists(src):
        raise HTTPException(400, "Pick an existing dataset (upload one first).")
    if not cfg.get("primarySparkHost"):
        raise HTTPException(400, "No Spark configured.")

    multinode = bool(cfg.get("useBothSparks") and cfg.get("secondarySparkHost"))
    run_id = _safe_id()
    local = _run_path(run_id)
    os.makedirs(local, exist_ok=True)

    # Assemble the run dir.
    with open(os.path.join(local, "dataset.jsonl"), "wb") as out, open(src, "rb") as inp:
        out.write(inp.read())
    with open(os.path.join(local, "dataset_info.json"), "w") as f:
        json.dump(lf_config.dataset_info("dataset.jsonl"), f, indent=2)
    with open(os.path.join(local, "config.yaml"), "w") as f:
        f.write(lf_config.train_yaml(cfg, multinode=multinode))
    if multinode:
        with open(os.path.join(local, "ds_z3.json"), "w") as f:
            f.write(lf_config.deepspeed_z3())

    sparks = sc.sparks(cfg)
    remote = _remote_run_dir(cfg, run_id)
    token = _hf_token()

    # Push the run dir to every participating Spark.
    for sp in sparks:
        r = sc.push_dir(sp, local, remote)
        if r.returncode != 0:
            raise HTTPException(502, f"rsync to {sp.host} failed: {r.stderr}")

    # Launch a detached container per node.
    image = cfg.get("dockerImage", "llamafactory-spark:latest")
    master = sparks[0].host
    for rank, sp in enumerate(sparks):
        env = [f"-e HF_TOKEN={shlex.quote(token)}"] if token else []
        if multinode:
            env += [
                "-e NNODES=2", f"-e NODE_RANK={rank}",
                f"-e MASTER_ADDR={shlex.quote(master)}", "-e MASTER_PORT=29500",
                "-e NPROC_PER_NODE=1",
            ]
        net = "--network host" if multinode else ""
        cmd = (
            f"docker rm -f lf-{run_id} >/dev/null 2>&1; "
            f"docker run -d --name lf-{run_id} --gpus all --shm-size=16g {net} "
            f"-v {shlex.quote(remote)}:/workspace/run "
            f"-w /workspace/run "
            f"--entrypoint llamafactory-cli "
            f"{' '.join(env)} "
            f"{shlex.quote(image)} train config.yaml"
        )
        res = sc.run(sp, cmd, timeout=120)
        if res.returncode != 0:
            raise HTTPException(502, f"docker run on {sp.host} failed: {res.stderr or res.stdout}")

    meta = {
        "run_id": run_id, "dataset": dataset, "model": cfg.get("model", "google/gemma-4-31B-it"),
        "finetuning_type": cfg.get("finetuningType", "lora"),
        "multinode": multinode, "hosts": [sp.host for sp in sparks],
        "remote_dir": remote, "started": time.time(), "status": "running",
    }
    _write_run_meta(run_id, meta)
    return {"ok": True, "run_id": run_id}


@app.get("/api/runs")
def list_runs():
    runs = []
    for run_id in sorted(os.listdir(RUNS_DIR), reverse=True):
        if not os.path.isdir(_run_path(run_id)):
            continue
        meta = _read_run_meta(run_id)
        runs.append(meta or {"run_id": run_id})
    return runs


@app.get("/api/runs/{run_id}/status")
def run_status(run_id: str):
    meta = _read_run_meta(run_id)
    if not meta:
        raise HTTPException(404, "no such run")
    cfg = _cfg()
    sp = sc.sparks(cfg)[0]
    res = sc.run(sp, f"docker inspect -f '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' lf-{run_id} 2>/dev/null",
                 timeout=30)
    state = (res.stdout or "").strip()
    if state:
        status, _, code = state.partition(" ")
        meta["status"] = "completed" if status == "exited" and code == "0" else (
            "failed" if status == "exited" else status)
        meta["exit_code"] = code
        _write_run_meta(run_id, meta)
    return {"status": meta.get("status"), "raw": state}


@app.get("/api/runs/{run_id}/log")
def run_log(run_id: str, follow: int = 1):
    _read_run_meta(run_id) or {}
    cfg = _cfg()
    sp = sc.sparks(cfg)[0]
    if follow:
        cmd = f"docker logs -f --tail 400 lf-{run_id}"
        return StreamingResponse(sc.stream(sp, cmd), media_type="text/plain")
    res = sc.run(sp, f"docker logs --tail 800 lf-{run_id}", timeout=60)
    return JSONResponse({"log": (res.stdout or "") + (res.stderr or "")})


@app.post("/api/runs/{run_id}/stop")
def run_stop(run_id: str):
    cfg = _cfg()
    for sp in sc.sparks(cfg):
        sc.run(sp, f"docker rm -f lf-{run_id}", timeout=60)
    meta = _read_run_meta(run_id)
    if meta:
        meta["status"] = "stopped"
        _write_run_meta(run_id, meta)
    return {"ok": True}


@app.post("/api/runs/{run_id}/fetch")
def run_fetch(run_id: str):
    cfg = _cfg()
    sp = sc.sparks(cfg)[0]
    remote = _remote_run_dir(cfg, run_id)
    local_out = os.path.join(_run_path(run_id), "output")
    r = sc.pull_dir(sp, os.path.join(remote, "output"), local_out)
    if r.returncode != 0:
        raise HTTPException(502, f"rsync back failed: {r.stderr}")
    return {"ok": True, "local_path": local_out}


@app.get("/healthz")
def healthz():
    return {"ok": True}
