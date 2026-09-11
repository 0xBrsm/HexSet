# Evaluation and fitting contract

Before launching evaluation, fitting, ablations, weight sweeps or candidate
validation, read `docs/evaluation.md`.

- Use native HexSet with the served offer/response/counter/pick protocol.
- Automatic clearing is research-only. Never use it for evaluation, fitting,
  policy adoption or measurements presented as served Heximax behavior.
- The current stock arena uses automatic clearing. Native engine identity,
  successful replay and a `trading=true` flag do not certify the protocol.
- Verify the served driver against actual served traces before a campaign,
  including enabled bot gates, automatic clearing disabled at game level,
  proposals, responses, execution, offer budgets and public activity updates.
- Freeze source, protocol, seeds and a bounded budget before launching. Do not
  automatically expand or rerun failed screens.
- Preserve incorrectly scoped historical artifacts and label their limits;
  do not relabel automatic-clearing records as served evaluations.

See `docs/readouts/trading-protocol-correction.md` for the withdrawn automatic
clearing adoption and outstanding served-driver checkpoint.
