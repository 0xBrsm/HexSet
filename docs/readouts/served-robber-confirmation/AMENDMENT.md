# Pre-confirmation driver repair

The initial six preflight attempts failed while serializing the first served
RoundNote: it is a NamedTuple, not a dataclass. No confirmation game launched
and no strength outcome was available. Preserve all six attempts and their
journals in failed-preflight.tar.gz; their container exited nonzero after
1.190 seconds.

Repair only note serialization with _asdict(). This supersedes the initial
plan's no-retry rule for this identified preflight implementation failure;
no failed game is silently replaced or counted as strength evidence. Charge
all six attempts to the unchanged 408-attempt / 900-second aggregate cap.
Use fresh preflight seed 734900000 (plus 100 for candidate checks) and fresh
confirmation seed 734000000. Reduce confirmation to 392 games (98 per seat).
Six failed attempts + six new checks + 392 confirmation + at most two reserved
adoption traces = 406 attempts. No further extension. The candidate, baseline,
protocol, win gate and all policy settings remain unchanged.
