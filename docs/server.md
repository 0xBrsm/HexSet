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
| `--board` | `spiral` | How boards are dealt: `spiral` (the rulebook's lettered spiral) or `random` (number discs in random order, 6 and 8 apart). A game journalled without a board mode resumes as `random` |
| `--runtime` | none | Module imported first, so the bots it registers can be seated and listed (repeatable) |
| `--device` | `cpu` | ONNX Runtime device; another value asks for `<DEVICE>ExecutionProvider` with CPU fallback |
| `--games-dir` | `$HEXSET_UI_GAMES_DIR` | Journal directory; an empty string disables journaling |
| `--forwarded-header` | none | The header a reverse proxy puts the client's address in (`CF-Connecting-IP`, `X-Forwarded-For`: its last entry); without it, the socket peer |
| `--games-per-period` | `100` | New games one client address may deal per rolling `--period-days`, through the page, the API and MCP alike, counted across restarts from the journals; `0` is no limit |
| `--period-days` | `30` | The window `--games-per-period` counts over |
| `--live-tables-per-ip` | `5` | Unfinished games one client address may have in play at once; `0` is no limit |
| `--limit-exempt` | `127.0.0.0/8`, `::1/128` | A client network under no per-address limit; repeatable, and naming any replaces the default |
| `--contact` | none | Where to ask for a larger allowance: sent with a period-limit 429 and with an `allowance` of ten or fewer games |
| `--ranker` | none | `MODULE:FUNCTION`, the function that makes the [rankings](#rankings); imported at start, so a name that does not import stops the server. Without it there are no rankings |
| `--rankings-min-games` | `10` | Ranking rows that played fewer games are not shown |
| `--ranked-bot` | none | The opponent the [ranked queue](api.md#ranked-play) seats in a table's one seat no player fills, as the opponent picker names it; it never takes two seats at a table. Without one, a ranked table waits for players to fill it |
| `--ranked-entrant` | none | A server bot, as the opponent picker names it, that plays [ranked games](api.md#ranked-play) itself: one at a time at each game type it is offered at, ranked as `bot:<name>`. Repeat for more; a name the server does not offer, or the `--ranked-bot`, stops it at start |
| `--api-keys` | none | A JSON file of API keys, `{"<key>": {"name": "<holder>", "games_per_period": <n>}}`: a deal sent with one on `X-HexSet-Key` counts against that key's `games_per_period` (`0` is no limit) instead of its address's `--games-per-period`. Read at start |

Behind a reverse proxy or tunnel, name its header with `--forwarded-header`,
or every request appears to come from the proxy: one address, sharing one set
of limits, and the journal and request log learn nothing about who played.
The header is believed from every request, so a client that reaches the
server without passing through the proxy can name any address it likes.
Serve only through the proxy if that matters.

Environment:

- `HEXSET_UI_MODELS_DIR`: the directory scanned for `*.onnx` opponents, on
  every request, each listed by the name its metadata declares
  ([onnx.md](onnx.md#serving-a-model)). Unset, it is `models/` beside `hexset/` in a source checkout
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

## Rankings

The server ranks nothing itself. Given `--ranker MODULE:FUNCTION`, it calls
that function with every ranked game in the journal directory that ended,
and serves what it returns. Only a table the ranked queue dealt
([api.md](api.md#ranked-play)) counts; a casual one never reaches the ranker.
The rankings are served as `GET /api/rankings` ([api.md](api.md#rankings)),
the MCP `rankings` tool, and a page at `/rankings` that the New game
dialog links. The module only has to be importable, from the installed
packages or `PYTHONPATH`.

```python
def rank(games: list[dict]) -> dict:
    return {"boards": [{
        "key": "duel", "title": "Duel", "note": "Two seats, 15 points.",
        "columns": [{"key": "rank", "label": "#"}, {"key": "rating", "label": "Rating"}],
        "rows": [{"name": "claude-opus-5-5", "id": "<client id>", "games": 40,
                  "rank": 1, "rating": 1612}],
    }]}
```

It is called on a thread of its own: once at start, and again after each
ranked game that ends, so it never holds up play. It is handed every ranked
game that ended each time, played out or timed out, oldest first by when it
ended, and keeps nothing between calls; a
changed ranker takes effect from a restart, over the whole history. A ranker
that raises, or returns anything but boards of this shape, is logged and the
last good rankings stay up. Journaling off, it is handed no games.

A game, as `hexset.server.rankings.finished_game` reads it from its
journal:

| Key | Meaning |
| --- | --- |
| `id`, `code` | The journal's game id and the table's code |
| `game_type`, `board_mode` | As dealt (`standard`, `duel-variant`; `spiral`, `random`) |
| `ranked` | Dealt by the ranked queue; always true for a game the ranker is handed |
| `imported` | Filed from a record (`journal_of`), not dealt by this server |
| `dealt_at`, `ended_at` | UTC times, `YYYY-MM-DDTHH:MM:SSZ` |
| `ended`, `timed_out` | `finished`, played out; or `timeout`, aborted when a move took longer than 60 seconds, `timed_out` then the seats it was waiting on. What a timeout is worth is the ranker's call |
| `turns`, `winner` | The game's turns and the winning seat, `null` for a timeout |
| `seats` | Every seat that took part; one closed before the first move did not |

Each seat:

| Key | Meaning |
| --- | --- |
| `seat`, `name` | The seat and its occupant's name at the end |
| `points`, `won` | True victory points at the end (at the abort, for a timeout), and whether it won |
| `left` | Left part-way through; its pieces and points stand |
| `bot`, `swapped` | `{"name", "spec"}` for a bot seat, as it was seated, with the `version` its profile declared then; `swapped` when another bot held it after play began |
| `client` | A person's or model's client record as journalled: `id` (null when it sent none), `kind`, the profile it claimed with (`name`, `version`, `owner`, `url`, `description`, those it sent), then `via`, `ip`, `ua` and an API key's holder `key` |

A board's `key` is unique and its `title` is text; `note` is optional. Rows
are shown in the order given. Each row is a `name`, an optional `id` and
`games`, and a value for any of the board's `columns`: text, a number,
true/false or null (a column of numbers, or of percentages written as text
such as `78%`, is right-aligned), and `pinned: true` for a rating the ranker holds
fixed, a scale's reference, which the page marks with a pin. `games` is how
many games it played: a row under
`--rankings-min-games` is not served, one with no `games` is never cut,
and it may also be a column.
A row may also carry `version`, `owner`, `url` (http or https) and
`description`; a row whose `id` has a profile, the latest one its client sent
([api.md](api.md#client-identity-and-seat-recovery)), or the name a seat it
held was last renamed to, is served with that profile's instead, and its `name` too when the profile has one. A row whose
`id` is `bot:<entry>` stands for a server bot, and is served with the profile
its preset declares ([guide.md](guide.md)), the version the server runs now.
A profile changing re-serves the board without ranking again. Only these are served. Anything else a row carries, an address included, is
dropped, so what the ranker is handed is not published unless it names it.
The page marks the row whose `id` is the visitor's own client id, and the MCP
tool the rows whose `id` is the caller's.

A board may also carry `pairs`, its rows head to head: each a pair of row ids
`a` and `b` (each pair once), the ranked `games` both played, the ones each
won (`a_wins`, `b_wins`, at most `games` together) and optionally
`expected`, the ranker's chance of `a` beating `b`, with `expected_low` and
`expected_high` around it for the range it gives. A pair naming a row not
served is dropped. On the page a click on a row puts it in a card above the
table with its columns and profile, and a click on a second compares the
two from these: their wins and games, each one's share of the games one of
them won at either end of a bar split at the expected share, lighter
across its range.

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
