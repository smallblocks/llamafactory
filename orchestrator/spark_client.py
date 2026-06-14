"""SSH/rsync helpers for driving LLaMA-Factory on remote DGX Sparks.

Shells out to the system `ssh`/`rsync` (installed in the image) rather than
using a Python SSH lib, so we get `docker logs -f` streaming for free and the
exact same behavior a human would get from a shell. Also usable as a CLI:

    python spark_client.py test     # probe GPU + image on the configured Spark(s)
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass

DATA_DIR = os.environ.get("LF_DATA_DIR", "/data")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
KEY_PATH = os.path.join(DATA_DIR, "ssh", "id_spark")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _ensure_key_perms() -> str:
    """SSH refuses world-readable keys. Copy to a private 600 path at runtime."""
    safe = "/tmp/id_spark"
    if not os.path.exists(KEY_PATH):
        raise FileNotFoundError(
            f"SSH key not found at {KEY_PATH}. Run the 'Configure Sparks' action first."
        )
    with open(KEY_PATH, "rb") as src, open(safe, "wb") as dst:
        dst.write(src.read())
    os.chmod(safe, 0o600)
    return safe


@dataclass
class Spark:
    host: str
    user: str
    port: int

    def ssh_base(self) -> list[str]:
        key = _ensure_key_perms()
        return [
            "ssh", "-i", key, "-p", str(self.port),
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15",
            f"{self.user}@{self.host}",
        ]


def sparks(cfg: dict | None = None) -> list[Spark]:
    cfg = cfg or load_config()
    user = cfg.get("primarySparkUser", "nvidia")
    port = int(cfg.get("sshPort", 22))
    out = [Spark(cfg["primarySparkHost"], user, port)]
    if cfg.get("useBothSparks") and cfg.get("secondarySparkHost"):
        out.append(Spark(cfg["secondarySparkHost"], user, port))
    return out


def run(spark: Spark, remote_cmd: str, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a shell command on the Spark, capturing output."""
    return subprocess.run(
        spark.ssh_base() + [remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def stream(spark: Spark, remote_cmd: str):
    """Yield stdout lines from a long-running remote command (e.g. docker logs -f)."""
    proc = subprocess.Popen(
        spark.ssh_base() + [remote_cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            yield line
    finally:
        proc.terminate()


def push_dir(spark: Spark, local_dir: str, remote_dir: str) -> subprocess.CompletedProcess:
    """rsync a local run dir up to the Spark."""
    key = _ensure_key_perms()
    ssh = f"ssh -i {key} -p {spark.port} -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
    run(spark, f"mkdir -p {shlex.quote(remote_dir)}")
    return subprocess.run(
        ["rsync", "-az", "--delete", "-e", ssh,
         local_dir.rstrip("/") + "/", f"{spark.user}@{spark.host}:{remote_dir.rstrip('/')}/"],
        capture_output=True, text=True,
    )


def pull_dir(spark: Spark, remote_dir: str, local_dir: str) -> subprocess.CompletedProcess:
    """rsync a remote output dir back down to the StartOS volume."""
    key = _ensure_key_perms()
    ssh = f"ssh -i {key} -p {spark.port} -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
    os.makedirs(local_dir, exist_ok=True)
    return subprocess.run(
        ["rsync", "-az", "-e", ssh,
         f"{spark.user}@{spark.host}:{remote_dir.rstrip('/')}/", local_dir.rstrip("/") + "/"],
        capture_output=True, text=True,
    )


def test_cli() -> int:
    cfg = load_config()
    if not cfg.get("primarySparkHost"):
        print("No Spark configured. Run the 'Configure Sparks' action first.")
        return 1
    image = cfg.get("dockerImage", "llamafactory-spark:latest")
    rc_all = 0
    for sp in sparks(cfg):
        print(f"== {sp.user}@{sp.host}:{sp.port} ==")
        r = run(sp, "nvidia-smi -L && echo '---' && "
                    f"(docker image inspect {shlex.quote(image)} >/dev/null 2>&1 "
                    f"&& echo 'image present: {image}' || echo 'image MISSING: build with spark/build.sh')",
                timeout=30)
        print(r.stdout.strip() or "(no output)")
        if r.returncode != 0:
            rc_all = r.returncode
            print(f"[error rc={r.returncode}] {r.stderr.strip()}")
        print()
    return rc_all


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "test":
        sys.exit(test_cli())
    print(f"unknown command: {cmd}")
    sys.exit(2)
