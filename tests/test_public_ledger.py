from types import SimpleNamespace

from hexset.catanatron.public_ledger import PublicResourceLedger, observe_public_action


def test_maritime_then_housekeeping_retains_known_lower_bound():
    b=PublicResourceLedger.new(2); b.initialise([[4,0,0,0,0],[0]*5],object())
    observe_public_action(b,seat=0,action_type='MARITIME_TRADE',value=('WOOD',None,None,None,'BRICK'))
    assert b.ledger.seats[0].known == [0,1,0,0,0]
    observe_public_action(b,seat=0,action_type='ROLL',value=7,hand_sizes=[3,0])
    assert b.ledger.seats[0].known == [0,1,0,0,0]


def test_unsupported_hidden_payload_is_metamorphic():
    states=[]
    for payload in ('WOOD','ORE',{'private':'anything'}):
        b=PublicResourceLedger.new(2); b.initialise([[0]*5,[0,0,0,0,2]],object())
        observe_public_action(b,seat=1,action_type='MOVE_ROBBER',value=payload,hand_sizes=[0,1])
        states.append([(x.known[:],x.unknown) for x in b.ledger.seats])
    assert states[0] == states[1] == states[2]


def test_history_cursor_replay_order_and_game_reset():
    b=PublicResourceLedger.new(2); token=object(); b.initialise([[1,0,0,0,0],[0]*5],token)
    events=[('BUILD_ROAD',None),('END_TURN',None),('ROLL',6)]
    for kind,value in events: observe_public_action(b,seat=0,action_type=kind,value=value)
    assert b.game_token is token
    b.reset(2,object())
    assert all(x.total()==0 for x in b.ledger.seats)


def test_randomized_seat_mapping_updates_named_seat_only():
    b=PublicResourceLedger.new(4); b.initialise([[0]*5 for _ in range(4)],object())
    observe_public_action(b,seat=3,action_type='MARITIME_TRADE',value=('ORE',None,None,None,'WOOD'))
    assert b.ledger.seats[3].known == [1,0,0,0,0]
    assert all(sum(x.known)==0 for x in b.ledger.seats[:3])


def test_dcp_current_roll_credits_visible_settlement_and_skips_seven_or_empty_bank():
    from hexset.catanatron.public_ledger import observe_public_roll
    book = PublicResourceLedger.new(2)
    book.initialise([[0]*5, [0]*5], object())
    tile = SimpleNamespace(number=6, resource='WOOD', nodes={7: 'node'})
    board = SimpleNamespace(
        map=SimpleNamespace(land_tiles={(1, 2, 3): tile}),
        buildings={'node': ('RED', 'SETTLEMENT')},
        robber_coordinate=None,
    )
    state = SimpleNamespace(board=board, colors=('RED','BLUE'), color_to_index={'RED': 0}, resource_freqdeck=[5, 5, 5, 5, 5])
    observe_public_roll(book, state=state, dice=(3, 3))
    assert book.ledger.seats[0].known[0] == 1
    observe_public_roll(book, state=state, dice=(6, 1))
    assert book.ledger.seats[0].known[0] == 1
    state.resource_freqdeck[0] = 0
    observe_public_roll(book, state=state, dice=(3, 3))
    assert book.ledger.seats[0].known[0] == 1


def test_dcp_skips_partial_payout_when_bank_has_fewer_than_visible_demand():
    from hexset.catanatron.public_ledger import observe_public_roll
    book = PublicResourceLedger.new(2)
    book.initialise([[0]*5, [0]*5], object())
    tile = SimpleNamespace(number=6, resource='WOOD', nodes={7: 'a', 8: 'b'})
    board = SimpleNamespace(
        map=SimpleNamespace(land_tiles={(1, 2, 3): tile}),
        buildings={
            'a': ('RED', 'SETTLEMENT'),
            'b': ('RED', 'SETTLEMENT'),
        },
        robber_coordinate=None,
    )
    # One bank card remains, but two settlements visibly request two.
    state = SimpleNamespace(board=board, colors=('RED','BLUE'), color_to_index={'RED': 0}, resource_freqdeck=[1, 5, 5, 5, 5])
    observe_public_roll(book, state=state, dice=(3, 3))
    assert book.ledger.seats[0].known == [0, 0, 0, 0, 0]


def test_dcp_aggregate_shortage_skips_all_seats_for_resource():
    from hexset.catanatron.public_ledger import observe_public_roll
    book = PublicResourceLedger.new(2); book.initialise([[0]*5,[0]*5], object())
    tile = SimpleNamespace(number=6, resource='WOOD', nodes={7: 'a', 8: 'b'})
    board = SimpleNamespace(map=SimpleNamespace(land_tiles={(1,2,3): tile}),
        buildings={'a': ('RED','SETTLEMENT'), 'b': ('BLUE','SETTLEMENT')}, robber_coordinate=None)
    state = SimpleNamespace(board=board, colors=('RED','BLUE'), color_to_index={'RED':0,'BLUE':1}, resource_freqdeck=[1,5,5,5,5])
    from hexset.catanatron.public_ledger import observe_public_roll
    observe_public_roll(book, state=state, dice=(3,3))
    assert [row.known[0] for row in book.ledger.seats] == [0,0]


def test_dcp_production_uses_randomized_seating_order():
    from hexset.catanatron.public_ledger import observe_public_roll
    book = PublicResourceLedger.new(2); book.initialise([[0]*5,[0]*5], object())
    tile = SimpleNamespace(number=6, resource='WOOD', nodes={7: 'node'})
    board = SimpleNamespace(map=SimpleNamespace(land_tiles={(1,2,3): tile}), buildings={'node': ('BLUE','SETTLEMENT')}, robber_coordinate=None)
    state = SimpleNamespace(board=board, colors=('ORANGE','BLUE'), color_to_index={'ORANGE':0,'BLUE':2}, resource_freqdeck=[5]*5)
    observe_public_roll(book, state=state, dice=(3,3))
    assert book.ledger.seats[1].known[0] == 1 and book.ledger.seats[0].known[0] == 0


def test_unsupported_event_always_clears_typed_certainty_without_totals():
    b = PublicResourceLedger.new(2); b.initialise([[0]*5, [0]*5], object())
    b.receive(1, 2, 2)
    observe_public_action(b, seat=1, action_type='DISCARD_RESOURCE', value='WOOD')
    assert b.ledger.seats[1].known == [0, 0, 0, 0, 0]
