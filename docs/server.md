# Running the server

`python -m hexset.server.web` serves HexSet over HTTP from one process: the
browser page, the JSON API and an MCP endpoint. Every participant acts
through a seat token, and embedded bots are clients of the same
`/api/action` route, so a submitted action returns the state after it and
later bot moves arrive as state changes.

The client interfaces are [api.md](api.md) for the JSON routes and
[mcp.md](mcp.md) for the MCP tools. Playing in a browser is covered in the
[README](../README.md#play-in-a-browser).

## Configuration

| Option | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Listening address; also accepted as an MCP `Origin` |
| `--port` | `8770` | Listening port |
| `--no-browser` | off | Do not open a browser tab at start |
| `--seed` | none | Deal every game from this seed (same board, same chance) |
| `--runtime` | none | Module imported first, so the bots it registers can be seated and listed (repeatable) |
| `--device` | `cpu` | ONNX Runtime device; another value asks for `<DEVICE>ExecutionProvider` with CPU fallback |
| `--games-dir` | `$HEXSET_UI_GAMES_DIR` | Journal directory; an empty string disables journaling |

Environment:

- `HEXSET_UI_MODELS_DIR`: the directory scanned for `*.onnx` opponents, on
  every request. Unset, it is `models/` beside `hexset/` in a source checkout
  (the directory holding `pyproject.toml`), else `models/` under the working
  directory.
- `HEXSET_UI_GAMES_DIR`: the journal directory when `--games-dir` is
  omitted; unset, `games` under the working directory; empty disables
  journaling.

## Docker

The image installs HexSet with the `server` and `catanatron` extras (ONNX
Runtime and the pinned Catanatron revision from `pyproject.toml`) and runs as
UID 10001. The example Compose file bind-mounts `hexset/` and `models/`
read-only over the installed copy and `games/` writable. Prepare the journal
directory before starting it:

```sh
cp compose.example.yaml compose.yaml
mkdir -p games
sudo chown 10001 games
docker compose up -d --build
```

Alternatively, set `user:` in `compose.yaml` to a user that owns `games/`.
Open `http://localhost:8770`; edit the port mapping in `compose.yaml` to
publish elsewhere.

`compose.yaml` is gitignored. Python source changes need
`docker compose restart`; `index.html` is read on each request. Dependency
or Dockerfile changes need a rebuild. The container runs with a read-only
root filesystem, a tmpfs `/tmp`, no capabilities and `no-new-privileges`.

## Saved games

The server writes one JSON Lines journal per game: every action, hidden cards
and chance outcomes included. A journal that cannot be written disables
journaling for that game with one logged line; play continues.

A game not in memory is rebuilt from its journal on the first request
for its code, after a restart or otherwise:

- an unfinished game comes back live and keeps journaling into the same file;
- a finished game comes back read-only;
- a game evicted after 24 hours untouched is closed, and its code is a 404;
- a journal that does not replay is closed, and its code is a 404.

Without journaling there is no recovery.

Rebuilt games keep their seats, names and client identities. Bot seats get
fresh tokens and resume. A person's seat comes back claimed and without a
token, so `/api/join` cannot hand it to someone else; its owner gets a new
token through `/api/reclaim`. The browser keeps its client secret and
reclaims on its own.

`journal_of(record, directory, *, code, names=None, notes=())` in
`hexset.server._journal` writes a replayable `hexset.record.Record` as a
journal filed under `code`, and `/<code>` then serves it like a game this
server dealt, replay included. Such a journal has no seed: its board and
chance outcomes are the recorded ones. `notes` are the record's trade
rounds as `(step, traded, RoundNote)`, each placed after the action at
`step` and the first `traded` exchanges filed on it.

## Seats and visibility

Before the first move an empty seat can be closed and reopened. After it the
seats are fixed. Leaving retires the caller's seat for good: its pieces and
cards stay, its turns are skipped, and nobody else can take it. A seat cannot
leave while a trade round it is part of is open.

A finished game is read-only for seats and spectators: every seat POST is a
409. A participant may still reclaim a finished seat to read it.

The token-free routes (`/api/table/<code>`, its `/board` and `/replay`)
need only the game code and show every hand, development card and true
victory-point count. A seat token gates actions and the seat's own view; it
does not stop a player from reading the spectator view. A seat reading a
replay with its token sees its own hand only, until the game is over, when
every hand is shown to every reader.
