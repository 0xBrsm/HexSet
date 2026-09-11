# Raw evidence

`stage4.tar.gz` contains all per-game records, summaries and preflight
outputs through manifest v4. Archived only after all five containers
completed. It contains 1,216 exploratory games and 24 preflight plays;
32-game summaries overlap the later 128-game summaries. Do not add their
counts together. Every game record carries full provenance and trace hash.

SHA256: `3ab0bc3f891582ce94a187d9020283ce4d30a47c68ac261a211c24a21d239baa`.

`stage6.tar.gz` supersedes stage4 for analysis: it adds the complete diagnostic
and initial development screens (2,624 exploratory games total), plus eight
new-source default-control preflight plays. SHA256: `44a904b727981db6055e92a11067521e50db54cda974e707842976e4e9ac8cc8`.

`stage8.tar.gz` adds the development extension, opening screen and initial
expansion screen: 3,648 exploratory games and 48 preflight plays total.
SHA256: `8e5f4930548b82bcb46916b9eb3ee8017928b8d54112af12245c30c5d3b85afd`.

`screens-complete.tar.gz` is the complete pre-confirmation evidence: 4,032
exploratory games and 48 preflight plays, including expansion025 at 256/gate.
SHA256: `f800eb0b639d5eba4b00f6b6f450dc347e79138697219b325fdfe6caea749baf`.

`confirmation-exp025.tar.gz` contains only the 1,024 fresh confirmation games
on seed 712000000. AB2 did not meet the registered advancement rule; shipped
win rate was 31.45% with a 95% lower bound of 27.57%. No final holdout.
SHA256: `cdc77227132442cee0cedfe89bcf40dc739cae49089d9d593b2db4f7883050ed`.

`probability-screen.tar.gz` contains 256 exploratory games and eight exact
default-control preflight replays for the optional probability-backup change.
SHA256: `ac970a959c54d5340a4731ffd00a10913540d4d7f67455d632a5ea4fa6b70392`.

`expansion-local-screen.tar.gz` contains the fresh local selection block,
seed 714000000: five arms at 64/gate, three extended to 256/gate (1,792
unique exploratory games). SHA256: `43035a8ea63a37b918e936a3f13958f80e38cb3c624fdaf87a8d5c6c2cf2d0d3`.
