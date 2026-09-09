# CPU image with a complete installed package. Compose overrides the source
# through a read-only bind mount for local development.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="HexSet"
LABEL org.opencontainers.image.source="https://github.com/0xBrsm/HexSet"

# Resolve dependencies and the pinned Catanatron revision from pyproject.toml.
COPY pyproject.toml LICENSE /tmp/hexset-build/
COPY src/ /tmp/hexset-build/src/
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && pip install --no-cache-dir '/tmp/hexset-build[server,catanatron]' \
    && apt-get purge -y --auto-remove git \
    && rm -rf /var/lib/apt/lists/* /tmp/hexset-build

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

USER hexset

# HOME points at what will be a tmpfs (see compose.example.yaml) because the
# root filesystem is read-only there: anything that goes looking for a
# writable home directory finds one instead of an error.
ENV HOME=/tmp
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
# Log lines reach `docker logs` as they happen rather than when a buffer
# happens to flush.
ENV PYTHONUNBUFFERED=1
ENV HEXSET_UI_MODELS_DIR=/app/models
# The one directory the app writes to: a full journal of every game, hidden
# cards and all, one file per game (see src/hexset/server/journal.py). Named
# here rather than left to the code's default so it cannot land relative to
# whatever WORKDIR happens to be. Set it empty to journal nothing; a directory
# that turns out not to be writable disables the journal with a line in
# `docker logs` rather than interrupting anyone's game.
ENV HEXSET_UI_GAMES_DIR=/app/games

WORKDIR /app
EXPOSE 8770
CMD ["python", "-m", "hexset.server.web", "--host", "0.0.0.0", "--port", "8770", "--no-browser"]
