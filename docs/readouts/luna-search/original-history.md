# Original Heximax history and bounded follow-up

This note separates the pre-campaign evidence from the September 10 stance
campaign. The original hand-value redesign landed in commit `3f5cc5f` and
changed `src/hexset/bots/evaluate.py` and
`src/hexset/bots/heximax/evaluate.py`. The play sweep and Catanatron readout
landed in commit `acd0575`, primarily in
`docs/readouts/heximax-fit/README.md`; the hand-valuation fit history landed
in `dc26938` under `docs/readouts/hand-valuation/`.

The original no-trade sweep varied production, buy progress, robber risk,
spare card, road, diversity, port, and knight one at a time. It adopted
robber risk `-0.30` and spare card `0.1065`; port and knight were effectively
inert at that resolution. The later 12-candidate stance screen does test some
joint vectors, so this is a bounded gap rather than an untested feature space:
the original one-term sweep did not establish interactions among terms.

The fair determinization follow-up holds per-world search budget constant:
depth 2, width 12, no-trade and `max_trades=0`, with `(k=1,max_nodes=2400)`
versus `(k=4,max_nodes=9600)`. `k` is determinized worlds, while
`max_nodes` is the total leaf budget (`src/hexset/bots/heximax/search.py`).
Use paired fresh seeds. The local 120-game timing and the live 1,024-game
AB2 timing differ by machine and workload; budget from the live report.
