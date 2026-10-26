# SPDX-License-Identifier: GPL-3.0-only
import itertools
import json

import pytest

from hexset.coalition import ActionIntent, Coalition, Event, ResourceIntent, TABLE, Transcript


def example():
    return Event("utterance-1", 2, 12, (
        Coalition(0, 2, 3, (ResourceIntent("ore", 2, 0, 2, "request"),),
                  (ActionIntent("build_city", 2, None, "statement"),)),
        Coalition(TABLE, None, 3)))


def test_round_trip_and_every_seat_frame_preserve_all_participants():
    event = example()
    assert Event.from_dict(json.loads(json.dumps(event.to_dict()))) == event
    for order in itertools.permutations(range(4)):
        framed = event.framed(order)
        assert framed.framed(tuple(order.index(s) for s in range(4))) == event
        assert framed.speaker == order.index(2)
        assert framed.symbols[0].resources[0].giver == order.index(0)
        assert framed.symbols[0].actions[0].actor == order.index(2)
        assert framed.symbols[1].addressed_to == TABLE


def test_reconnect_and_multiple_symbols_do_not_inflate_counts():
    log = Transcript()
    assert log.append(example())
    assert not log.append(example())
    assert log.against == {(2, 3): 1}
    log.append(Event("utterance-2", 2, 13, (Coalition(opposed=3), Coalition(opposed=0))))
    assert log.against == {(2, 3): 2, (2, 0): 1}
    external = log.against
    external.clear()
    assert log.against
    with pytest.raises(ValueError, match="different content"):
        log.append(Event("utterance-1", 2, 12, ()))


def test_causal_prefix_excludes_same_position_and_requires_known_order():
    log = Transcript()
    log.append(example())
    assert log.before(12) == ()
    assert log.before(13) == (example(),)
    with pytest.raises(ValueError, match="nondecreasing"):
        log.append(Event("earlier", 0, 11, ()))
    log.append(Event("unknown", 0, None, ()))
    with pytest.raises(ValueError, match="unresolved"):
        log.before(13)


def test_unknown_is_distinct_from_table_and_zero_from_unknown_quantity():
    assert Coalition().addressed_to is None
    assert Coalition(addressed_to=TABLE).addressed_to == TABLE
    assert ResourceIntent("ore", 0) != ResourceIntent("ore")
    for factory in (lambda: ResourceIntent("dev"), lambda: ResourceIntent("ore", -1),
                    lambda: ActionIntent("invented"), lambda: Coalition(opposed=-2),
                    lambda: example().framed((0, 1, 2))):
        with pytest.raises(ValueError):
            factory()
