# A small CPU-only image: onnxruntime inference, no GPU toolchain, and no
# build step for the frontend (it is one static HTML file).
#
# Dependencies only — src/, the engine and models/ are bind-mounted at run
# time rather than copied in (see compose.example.yaml), so editing the code
# is a restart and not a rebuild; only a dependency bump touches the image.
# If you would rather have a self-contained image to push to a registry, add
#
#     COPY src/ /app/src/
#
# below and drop the matching volume from your compose file.
#
# The engine is a separate mount because it is a separate repository: the
# rules, the bots and the ONNX record contract are the `hexset` package
# (0xBrsm/dev-hexset, src/), and this repo is the gym around it. Until that
# package is on an index it is bind-mounted at /app/engine and reached over
# PYTHONPATH, exactly as src/ is.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="HexSet-UI"
LABEL org.opencontainers.image.source="https://github.com/0xBrsm/HexSet-UI"

# Kept in step with pyproject.toml's [project].dependencies by hand. A version
# bump belongs in both places or `pip install -e .` and this image drift
# apart. `hexset` is not installed here: it is not on an index yet, so it
# arrives as the /app/engine bind mount below. Its own only dependency is
# numpy, which is already here.
RUN pip install --no-cache-dir "numpy>=1.26" "onnxruntime>=1.18"

# Runs unprivileged. The uid is arbitrary and deliberately high enough not to
# collide with a host account: it reads the bind-mounted source and models,
# both mounted read-only, and writes only to the games directory below. That
# one has to be writable *by this uid* on the host — `chown 10001` it, or set
# `user:` in your compose file to something your host already allows.
RUN useradd --system --no-create-home --uid 10001 hexset

# Created before the drop to an unprivileged user so it exists (and is owned)
# even when nothing is mounted over it: with the root filesystem read-only,
# the app could not otherwise make it itself, and every game would go
# unjournalled on a fresh `docker run` with no compose file.
RUN mkdir -p /app/games && chown hexset /app/games

# Created for the same reason as /app/games: so a `docker run` with nothing
# mounted over it fails with an import error naming `hexset` rather than a
# read-only-filesystem error naming nothing.
RUN mkdir -p /app/engine

USER hexset

# HOME points at what will be a tmpfs (see compose.example.yaml) because the
# root filesystem is read-only there: anything that goes looking for a
# writable home directory finds one instead of an error.
ENV HOME=/tmp
# src/ is this repo; engine/ is dev-hexset's src/, holding the `hexset`
# package. Order matters only in that both must be present.
ENV PYTHONPATH=/app/src:/app/engine
ENV PYTHONDONTWRITEBYTECODE=1
# Log lines reach `docker logs` as they happen rather than when a buffer
# happens to flush.
ENV PYTHONUNBUFFERED=1
ENV HEXSET_UI_MODELS_DIR=/app/models
# The one directory the app writes to: a full journal of every game, hidden
# cards and all, one file per game (see src/hexset_ui/journal.py). Named here
# rather than left to the code's default so it cannot land relative to
# whatever WORKDIR happens to be. Set it empty to journal nothing; a directory
# that turns out not to be writable disables the journal with a line in
# `docker logs` rather than interrupting anyone's game.
ENV HEXSET_UI_GAMES_DIR=/app/games

WORKDIR /app
EXPOSE 8770
CMD ["python", "-m", "hexset_ui.web", "--host", "0.0.0.0", "--port", "8770", "--no-browser"]
