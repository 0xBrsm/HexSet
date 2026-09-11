# Evaluation and fitting contract

Before launching evaluation, fitting, ablations, weight sweeps or candidate
validation, read `docs/evaluation.md`.

- Use native HexSet with the served offer/response/counter/pick protocol.
- Automatic clearing is research-only. Never use it for evaluation, fitting,
  policy adoption or measurements presented as served Heximax behavior.
- The stock arena defaults to engine-driven offer/response rounds. This alone
  does not certify served equivalence: budgets, proposal policy and public
  activity notifications must match the intended served implementation.
  Native engine identity, replay and `trading=true` are insufficient.
- Verify the served driver against actual served traces before a campaign,
  including enabled bot gates, `trade_mode="external"` for session-driven games,
  proposals, responses, execution, offer budgets and public activity updates.
- Freeze source, protocol, seeds and a bounded budget before launching. Do not
  automatically expand or rerun failed screens.
- Preserve incorrectly scoped historical artifacts and label their limits;
  do not relabel automatic-clearing records as served evaluations.

See `docs/readouts/trading-protocol-correction.md` for the withdrawn automatic
clearing adoption. The archived served confirmation and its current scope are
in `docs/readouts/served-robber-confirmation/CURRENT-SCOPE.md`.
