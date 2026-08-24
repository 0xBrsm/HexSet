# catan-web:local — CPU-only, onnxruntime inference, no GPU toolchain.
# Contrast with dev-catan's own docker/Dockerfile: that image exists almost
# entirely to bridge ROCm to a GPU (a pinned librocdxg fetch, HSA/hipBLASLt
# env vars, a multi-GB PyTorch base) — none of it applies here. This is a
# small, static, single-stage build meant to run comfortably on a plain CPU
# box with a few GB of RAM (e.g. no discrete GPU at all).
FROM python:3.12-slim

LABEL org.opencontainers.image.title="catan-web"
LABEL org.opencontainers.image.source="https://github.com/0xBrsm/catan-web"

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Models are bind-mounted or copied in separately at run time, never baked
# into the image — "drop a file in, it shows up in the picker" is the whole
# point of this repo, and that has to work without a rebuild.
VOLUME /app/models
ENV CATAN_WEB_MODELS_DIR=/app/models

EXPOSE 8770
CMD ["python", "-m", "catan.webserver", "--host", "0.0.0.0", "--port", "8770", "--no-browser"]
