from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum

from .board.board import MAX_ROLL, MIN_ROLL, Board, pips
from .board.terrain import TERRAIN_RESOURCE, Resource
from .cards import ROAD_BUILDING_ROADS, DevCard
from .devcards import buy as buy_dev_card
from .devcards import (
    mature,
    play_knight,
    play_monopoly,
    play_year_of_plenty,
    spend_card,
)
from .economy import Purchase, bank_trade, distribute, pay
from .ledger import PublicLedger
from .robber import discard_count, move_robber, steal
from .trading import Offer, can_propose
from .trading import execute as execute_trade
from .trading import responders as offer_responders
from .state import (
    NO_OWNER,
    GameState,
    copy_state,
    new_game,
    place_road,
    place_settlement,
    upgrade_to_city,
)
from .victory import update_largest_army, update_longest_road, winner

DICE = 6
MAX_TURNS = 1000

# Offers are uncapped in composition — a player may ask for anything in exchange
# for anything they hold. This caps how many *separate* offers one turn may
# contain, and is a termination guarantee rather than a rule: the turn counter
# only advances on `end_turn`, so a player who negotiated forever would never
# reach `MAX_TURNS`. Humans stop because their opponents lose patience.
MAX_OFFERS_PER_TURN = 8

ROLL_ODDS: tuple[tuple[int, float], ...] = tuple(
    (roll, pips(roll) / DICE**2) for roll in range(MIN_ROLL, MAX_ROLL + 1)
)


class Phase(IntEnum):
    SETUP_SETTLEMENT = 0
    SETUP_ROAD = 1
    ROLL = 2
    DISCARD = 3
    ROBBER = 4
    MAIN = 5
    GAME_OVER = 6
    TRADE_RESPOND = 7


@dataclass
class Game:
    state: GameState
    rng: random.Random
    ledger: PublicLedger
    phase: Phase = Phase.SETUP_SETTLEMENT
    current_player: int = 0
    setup_queue: list[int] = field(default_factory=list)
    setup_step: int = 0
    last_settlement: int = -1
    last_roll: int | None = None
    dev_card_played: bool = False
    discard_quota: list[int] = field(default_factory=list)
    free_roads: int = 0
    turns: int = 0
    won_by: int | None = None
    offer: Offer | None = None
    pending_responders: list[int] = field(default_factory=list)
    offers_made: int = 0


def start(
    board: Board, num_players: int, rng: random.Random | None = None
) -> Game:
    rng = rng or random.Random()
    order = list(range(num_players))
    # Snake order: the last player to place first also places first in round two,
    # which is what compensates them for choosing last.
    queue = order + order[::-1]
    return Game(
        state=new_game(board, num_players, rng),
        rng=rng,
        ledger=PublicLedger.new(num_players),
        setup_queue=queue,
        current_player=queue[0],
        discard_quota=[0] * num_players,
    )


def imagine(
    game: Game, rng: random.Random, *, randomize_deck: bool = True
) -> Game:
    """A copy for hypothetical play, safe to mutate and to draw from.

    Two things keep a search honest. It gets its own `rng`, so exploring
    branches cannot disturb the real game's random stream and leave the result
    unreproducible. And the copied deck is shuffled by default, so a search that
    buys a development card cannot read the card the real deck is about to deal.
    A caller that cannot observe the deck before a later draw may defer that
    shuffle until the draw itself with `randomize_deck=False`.
    """
    state = copy_state(game.state)
    if randomize_deck:
        rng.shuffle(state.deck)
    return Game(
        state=state,
        rng=rng,
        ledger=game.ledger.copy(),
        phase=game.phase,
        current_player=game.current_player,
        setup_queue=game.setup_queue[:],
        setup_step=game.setup_step,
        last_settlement=game.last_settlement,
        last_roll=game.last_roll,
        dev_card_played=game.dev_card_played,
        discard_quota=game.discard_quota[:],
        free_roads=game.free_roads,
        turns=game.turns,
        won_by=game.won_by,
        offer=game.offer,
        pending_responders=game.pending_responders[:],
        offers_made=game.offers_made,
    )


def _require(game: Game, phase: Phase) -> None:
    if game.phase is not phase:
        raise ValueError(f"expected phase {phase.name}, got {game.phase.name}")


def _in_second_setup_round(game: Game) -> bool:
    return game.setup_step >= game.state.num_players


def _snapshot_hands(game: Game) -> list[list[int]]:
    """A copy of every seat's hand, to diff against after a mutation whose
    resource identities are public (see `ledger.PublicLedger.apply_hand_diff`
    for which events these are and why a steal is never one of them)."""
    return [hand[:] for hand in game.state.hands]


def _record_steal(
    game: Game, thief: int, victim: int, stolen: Resource | None
) -> None:
    """The one hand mutation `_snapshot_hands`/`apply_hand_diff` must never
    see: a robber or knight steal moves one card whose identity is public to
    nobody but the thief and the victim. `stolen` is `robber.steal`'s own
    return value, read only for its `None`-ness (the victim held nothing to
    take, so nothing happened and there is nothing to record) -- never for
    the resource it names, which `ledger.PublicLedger.steal` must not be
    told: see its docstring for the identity-independent convention that
    keeps a steal's outcome unreadable from the encoded ledger, for every
    seat including the thief and the victim's neighbours."""
    if stolen is None:
        return
    game.ledger.steal(thief, victim)


def _grant_initial_resources(game: Game, vertex: int) -> None:
    state = game.state
    topology = state.board.topology
    for h in topology.vertex_hexes[vertex]:
        resource = TERRAIN_RESOURCE[state.board.terrain[h]]
        if resource is not None and state.bank[resource] > 0:
            state.bank[resource] -= 1
            state.hands[game.current_player][resource] += 1


def place_initial_settlement(game: Game, vertex: int) -> None:
    _require(game, Phase.SETUP_SETTLEMENT)
    before = _snapshot_hands(game)
    place_settlement(game.state, game.current_player, vertex, connected=False)
    game.last_settlement = vertex
    if _in_second_setup_round(game):
        _grant_initial_resources(game, vertex)
    game.ledger.apply_hand_diff(before, game.state.hands)
    update_longest_road(game.state)
    game.phase = Phase.SETUP_ROAD


def legal_initial_roads(game: Game) -> list[int]:
    topology = game.state.board.topology
    return [
        e
        for e in topology.vertex_edges[game.last_settlement]
        if game.state.edge_owner[e] == NO_OWNER
    ]


def place_initial_road(game: Game, edge: int) -> None:
    _require(game, Phase.SETUP_ROAD)
    if edge not in legal_initial_roads(game):
        raise ValueError("the opening road must touch the settlement just placed")
    place_road(game.state, game.current_player, edge)
    update_longest_road(game.state)

    game.setup_step += 1
    if game.setup_step < len(game.setup_queue):
        game.current_player = game.setup_queue[game.setup_step]
        game.phase = Phase.SETUP_SETTLEMENT
    else:
        game.current_player = 0
        game.phase = Phase.ROLL


def roll_dice(game: Game, roll: int | None = None) -> int:
    """Roll, or resolve a given roll so a search can enumerate the outcomes."""
    _require(game, Phase.ROLL)
    if roll is None:
        roll = game.rng.randint(1, DICE) + game.rng.randint(1, DICE)
    game.last_roll = roll

    if roll == 7:
        # Quotas are fixed now rather than recomputed as hands shrink, so
        # discarding does not reduce what is still owed.
        game.discard_quota = [
            discard_count(game.state, p) for p in range(game.state.num_players)
        ]
        game.phase = Phase.DISCARD if any(game.discard_quota) else Phase.ROBBER
    else:
        before = _snapshot_hands(game)
        distribute(game.state, roll)
        game.ledger.apply_hand_diff(before, game.state.hands)
        game.phase = Phase.MAIN
    return roll


def players_owing_discards(game: Game) -> list[int]:
    return [p for p, owed in enumerate(game.discard_quota) if owed > 0]


def to_move(game: Game) -> int:
    """Whose decision the legal actions belong to.

    Usually the player whose turn it is, but two phases hand the decision to
    somebody else: discarding on a seven is decided by whoever owes cards, and
    a trade offer is decided by whoever is being asked to take it.
    """
    if game.phase is Phase.DISCARD:
        owing = players_owing_discards(game)
        if owing:
            return owing[0]
    if game.phase is Phase.TRADE_RESPOND and game.pending_responders:
        return game.pending_responders[0]
    return game.current_player


def _finish_discards(game: Game) -> None:
    if not any(game.discard_quota):
        game.phase = Phase.ROBBER


def discard_one(game: Game, player: int, resource: Resource) -> None:
    _require(game, Phase.DISCARD)
    if game.discard_quota[player] < 1:
        raise ValueError(f"player {player} owes no discard")
    if game.state.hands[player][resource] < 1:
        raise ValueError(f"player {player} holds no {resource.name}")
    game.state.hands[player][resource] -= 1
    game.state.bank[resource] += 1
    game.ledger.spend(player, int(resource), 1)
    game.discard_quota[player] -= 1
    _finish_discards(game)


def move_robber_to(game: Game, target: int, victim: int | None = None) -> None:
    _require(game, Phase.ROBBER)
    move_robber(game.state, target)
    if victim is not None:
        stolen = steal(game.state, game.current_player, victim, game.rng)
        _record_steal(game, game.current_player, victim, stolen)
    game.phase = Phase.MAIN


def _check_win(game: Game) -> None:
    won = winner(game.state)
    if won is not None:
        game.won_by = won
        game.phase = Phase.GAME_OVER


def build_road(game: Game, edge: int) -> None:
    """Build a road, spending a free road from road building if one is owed."""
    _require(game, Phase.MAIN)
    before = _snapshot_hands(game)
    if game.free_roads > 0:
        game.free_roads -= 1
    else:
        pay(game.state, game.current_player, Purchase.ROAD)
    game.ledger.apply_hand_diff(before, game.state.hands)
    place_road(game.state, game.current_player, edge)
    update_longest_road(game.state)
    _check_win(game)


def build_settlement(game: Game, vertex: int) -> None:
    _require(game, Phase.MAIN)
    before = _snapshot_hands(game)
    pay(game.state, game.current_player, Purchase.SETTLEMENT)
    game.ledger.apply_hand_diff(before, game.state.hands)
    place_settlement(game.state, game.current_player, vertex)
    # A new settlement can cut an opponent's route, so this is not only the
    # builder's own longest road that may change.
    update_longest_road(game.state)
    _check_win(game)


def build_city(game: Game, vertex: int) -> None:
    _require(game, Phase.MAIN)
    before = _snapshot_hands(game)
    pay(game.state, game.current_player, Purchase.CITY)
    game.ledger.apply_hand_diff(before, game.state.hands)
    upgrade_to_city(game.state, game.current_player, vertex)
    _check_win(game)


def buy_development_card(game: Game) -> DevCard:
    _require(game, Phase.MAIN)
    before = _snapshot_hands(game)
    card = buy_dev_card(game.state, game.current_player)
    game.ledger.apply_hand_diff(before, game.state.hands)
    _check_win(game)
    return card


def _spend_turn_card(game: Game) -> None:
    if game.dev_card_played:
        raise ValueError("only one development card may be played per turn")
    game.dev_card_played = True


def play_knight_card(
    game: Game, target: int, victim: int | None = None
) -> Resource | None:
    if game.phase not in (Phase.ROLL, Phase.MAIN):
        raise ValueError(f"cannot play a knight in {game.phase.name}")
    _spend_turn_card(game)
    stolen = play_knight(game.state, game.current_player, target, victim, game.rng)
    if victim is not None:
        _record_steal(game, game.current_player, victim, stolen)
    update_largest_army(game.state)
    _check_win(game)
    return stolen


def play_road_building_card(game: Game) -> None:
    """Credit two free roads, placed afterwards with ordinary build actions.

    Resolving the card this way keeps the action space flat: one entry for the
    card, rather than one per pair of edges.
    """
    _require(game, Phase.MAIN)
    _spend_turn_card(game)
    spend_card(game.state, game.current_player, DevCard.ROAD_BUILDING)
    game.free_roads += ROAD_BUILDING_ROADS


def play_year_of_plenty_card(game: Game, resources: list[Resource]) -> None:
    _require(game, Phase.MAIN)
    _spend_turn_card(game)
    before = _snapshot_hands(game)
    play_year_of_plenty(game.state, game.current_player, resources)
    game.ledger.apply_hand_diff(before, game.state.hands)


def play_monopoly_card(game: Game, resource: Resource) -> int:
    _require(game, Phase.MAIN)
    _spend_turn_card(game)
    before = _snapshot_hands(game)
    taken = play_monopoly(game.state, game.current_player, resource)
    # Monopoly forces every other seat to publicly hand over every card of
    # `resource`, so the transfer is fully public despite touching every
    # seat at once -- the same `apply_hand_diff` every other public mutation
    # uses, not the hidden-identity path a steal needs.
    game.ledger.apply_hand_diff(before, game.state.hands)
    return taken


def trade_with_bank(game: Game, give: Resource, receive: Resource) -> None:
    _require(game, Phase.MAIN)
    before = _snapshot_hands(game)
    bank_trade(game.state, game.current_player, give, receive)
    game.ledger.apply_hand_diff(before, game.state.hands)


def propose_trade(
    game: Game,
    give: tuple[int, ...],
    want: tuple[int, ...],
    ask: tuple[int, ...] = (),
) -> None:
    """Put an offer to the table. Nobody is obliged to take it.

    `ask` is the proposer's order of preference, best first. It only reorders
    who is asked — it cannot add a player who could not cover the offer, and
    anyone left out of it keeps the engine's neutral order behind those named.
    """
    _require(game, Phase.MAIN)
    if game.offers_made >= MAX_OFFERS_PER_TURN:
        raise ValueError(f"only {MAX_OFFERS_PER_TURN} offers allowed per turn")

    offer = Offer(proposer=game.current_player, give=give, want=want)
    if not can_propose(game.state, offer):
        raise ValueError(f"player {game.current_player} cannot make this offer")

    game.offers_made += 1
    willing = offer_responders(game.state, offer)
    if not willing:
        # Nobody can cover it, so there is nothing to ask. The offer still
        # counts against the turn's allowance: it was a move that was made.
        return

    if ask:
        rank = {p: i for i, p in enumerate(ask)}
        willing = tuple(sorted(willing, key=lambda p: rank.get(p, len(ask))))

    game.offer = offer
    game.pending_responders = list(willing)
    game.phase = Phase.TRADE_RESPOND


def _finish_offer(game: Game) -> None:
    game.offer = None
    game.pending_responders = []
    game.phase = Phase.MAIN


def accept_trade(game: Game, responder: int) -> None:
    _require(game, Phase.TRADE_RESPOND)
    if not game.pending_responders or game.pending_responders[0] != responder:
        raise ValueError(f"player {responder} is not the one being asked")
    assert game.offer is not None
    before = _snapshot_hands(game)
    execute_trade(game.state, game.offer, responder)
    game.ledger.apply_hand_diff(before, game.state.hands)
    _finish_offer(game)


def decline_trade(game: Game, responder: int) -> None:
    _require(game, Phase.TRADE_RESPOND)
    if not game.pending_responders or game.pending_responders[0] != responder:
        raise ValueError(f"player {responder} is not the one being asked")
    game.pending_responders.pop(0)
    if not game.pending_responders:
        _finish_offer(game)


def end_turn(game: Game) -> None:
    _require(game, Phase.MAIN)
    mature(game.state, game.current_player)
    game.dev_card_played = False
    game.offers_made = 0
    game.offer = None
    game.pending_responders = []
    # Free roads with nowhere legal to go are simply lost.
    game.free_roads = 0
    game.turns += 1
    if game.turns >= MAX_TURNS:
        game.phase = Phase.GAME_OVER
        return
    game.current_player = (game.current_player + 1) % game.state.num_players
    game.phase = Phase.ROLL


def is_over(game: Game) -> bool:
    return game.phase is Phase.GAME_OVER
