# StartOS service image: the LLaMA-Factory orchestrator (control plane).
# Build context is the PROJECT ROOT (manifest images.main.workdir = '.'), so
# COPY paths are repo-relative. Arch-agnostic and GPU-free — it only SSHes/rsyncs
# to the Sparks. start-cli builds this for x86_64 + aarch64 (manifest arch).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-client rsync tini ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orchestrator/ ./

EXPOSE 8080
ENV LF_DATA_DIR=/data
# main.ts overrides this command, but keep a sane default for local runs.
ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
