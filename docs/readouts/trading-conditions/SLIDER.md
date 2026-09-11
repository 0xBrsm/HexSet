# One Heximax policy between the current best endpoints

Use `heximax-adaptive`, or `heximax(board, rng, adaptive=True)`:

    move_weights = (1 - activity) * NO_TRADE_WEIGHTS + activity * BALANCED_WEIGHTS
    expansion_credit = (1 - activity) * .25 + activity * .125

Activity zero is exactly the current no-trade move profile. Activity one is
exactly the best validated trading policy, `heximax-balanced`. That trading
endpoint is the newly confirmed midpoint M, not the legacy T move profile.
All intermediate coefficients are linear interpolations, with no new weight
fit, boundary offsets or minimum distance from an endpoint.

The exchange evaluator stays TRADING_WEIGHTS, without expansion credit, and
the floor is zero. These are the exchange settings of the best trading policy.
Zero activity changes the move weights; it does not disable trading and prevent
the bot from discovering that exchanges have become available. An explicit
bot or game trade-off switch selects the no-trade endpoint immediately.

Activity is the fraction of the last eight eligible public turns with a
completed exchange involving that turn's actor. Start with eight zero entries;
new observations replace them, so there is no permanent prior. Eight trading
observations reach one; eight quiet observations reach zero. Each turn has
one vote, regardless of the number of exchanges. Eligibility uses only public
hand sizes (at least two cards for the actor and some partner); a turn without
that coarse opportunity is skipped. This measures realized exchange activity,
not hidden willingness to accept offers.

The native engine publishes the public turn, actor, pre-exchange hand sizes
and completed participant pairs after clearing its once-per-turn event.
Hypothetical search games carry no seated gates and cannot update the observer.
The bot updates weights at the next decision and holds them fixed throughout
search, clearing all move-evaluation caches normally. Exchange values use their
separate evaluator and do not change while an event is clearing.

The fixed presets remain reproducible controls; users need not select a
separate profile based on opponents. Existing named presets retain their old
behavior rather than silently changing historical benchmark identities.

Verification covers exact endpoints, linear intermediate weights, both
activity transitions, deduplication, opportunity filtering, public-only event
payloads, search-copy isolation, fixed-profile move/RNG/gate parity, and the
explicit trade-off switch. `verify_slider.py` checks four full native no-trade
pairs across all candidate seats and four completed trading games with a
moving slider. This is a small behavior check, not a strength experiment. All four full-game
no-trade pairs matched exactly; all four trading games completed with activity
changes. The full default suite passed 868 tests (14 skipped, one deselected).

The earlier 520/1,536 adaptive result used a different interpolation (N to
legacy T), prior and observation memory. It motivated this work but must not
be reported as this revised policy's win rate. Frozen scripts and data remain
unchanged. No new broad ablation sweep is needed to check interpolation.
