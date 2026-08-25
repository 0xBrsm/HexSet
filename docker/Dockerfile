# A small CPU-only image: onnxruntime inference, no GPU toolchain, and no
# build step for the frontend (it is one static HTML file).
#
# Dependencies only — src/ and models/ are bind-mounted at run time rather
# than copied in (see compose.example.yaml), so editing the code is a restart
# and not a rebuild; only a dependency bump touches the image. If you would
# rather have a self-contained image to push to a registry, add
#
#     COPY src/ /app/src/
#
# below and drop the matching volume from your compose file.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="catan-web"
LABEL org.opencontainers.image.source="https://github.com/0xBrsm/catan-web"

# Kept in step with pyproject.toml's [project].dependencies by hand. There
# are two of them and they move rarely, but a version bump belongs in both
# places or `pip install -e .` and this image drift apart.
RUN pip install --no-cache-dir "numpy>=1.26" "onnxruntime>=1.18"

# Runs unprivileged. The uid is arbitrary and deliberately high enough not to
# collide with a host account: it only has to *read* the bind-mounted source
# and models, both of which are mounted read-only. If your host needs a
# particular uid:gid — because you added a writable mount, say — set `user:`
# in your compose file rather than editing this.
RUN useradd --system --no-create-home --uid 10001 catan
USER catan

# HOME points at what will be a tmpfs (see compose.example.yaml) because the
# root filesystem is read-only there: anything that goes looking for a
# writable home directory finds one instead of an error.
ENV HOME=/tmp
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
# Log lines reach `docker logs` as they happen rather than when a buffer
# happens to flush.
ENV PYTHONUNBUFFERED=1
ENV CATAN_WEB_MODELS_DIR=/app/models

WORKDIR /app
EXPOSE 8770
CMD ["python", "-m", "catan.webserver", "--host", "0.0.0.0", "--port", "8770", "--no-browser"]
