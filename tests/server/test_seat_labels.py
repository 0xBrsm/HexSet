"""Player numbers skip closed seats."""

from hexset.server.webplay import SeatLabels, _who


def test_numbers_skip_closed_seats():
    labels = SeatLabels({0: "heximax", 2: "human", 3: "search2"}, locked=[1])
    assert [labels.number(s) for s in (0, 2, 3)] == [1, 2, 3]
    assert _who(3, labels) == "Player 3 (search2)"


def test_plain_dict_keeps_seat_numbers():
    assert _who(3, {3: "search2"}) == "Player 4 (search2)"
