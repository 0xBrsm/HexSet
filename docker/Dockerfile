# catan-web:local — CPU-only, onnxruntime inference, no GPU toolchain.
# Contrast with dev-catan's own docker/Dockerfile: that image exists almost
# entirely to bridge ROCm to a GPU (a pinned librocdxg fetch, HSA/hipBLASLt
# env vars, a multi-GB PyTorch base) — none of it applies here.
#
# Dependencies only — src/ and models/ are bind-mounted at run time (see
# compose.yaml), not baked in. That means a code edit is a `git pull` +
# restart, not a rebuild; only a dependency-version bump touches the image.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="catan-web"
LABEL org.opencontainers.image.source="https://github.com/0xBrsm/catan-web"

RUN pip install --no-cache-dir "numpy>=1.26" "onnxruntime>=1.18"

ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV CATAN_WEB_MODELS_DIR=/app/models

WORKDIR /app
EXPOSE 8770
CMD ["python", "-m", "catan.webserver", "--host", "0.0.0.0", "--port", "8770", "--no-browser"]
