from activity import TradeActivity


def test_one_turn_is_one_observation_even_with_many_deals():
    a=TradeActivity();b=TradeActivity()
    a.observe(turn=1,actor=0,hand_sizes=(4,4,4,4),trade_participants=[(0,1)])
    b.observe(turn=1,actor=0,hand_sizes=(4,4,4,4),trade_participants=[(0,1)]*20)
    assert a==b and a.eligible_turns==1
    a.observe(turn=1,actor=0,hand_sizes=(4,4,4,4),trade_participants=[])
    assert a==b


def test_no_opportunity_does_not_count_as_refusal():
    a=TradeActivity()
    for turn,sizes in enumerate([(1,4,4,4),(4,1,1,1)]):
        a.observe(turn=turn,actor=0,hand_sizes=sizes,trade_participants=[])
    assert a.eligible_turns==0 and a.mean==.25


def test_recency_tracks_changing_public_conditions_and_stays_bounded():
    a=TradeActivity()
    for turn in range(64):a.observe(turn=turn,actor=0,hand_sizes=(4,4,4,4),trade_participants=[])
    quiet=a.mean
    for turn in range(64,96):a.observe(turn=turn,actor=0,hand_sizes=(4,4,4,4),trade_participants=[(0,1)])
    assert 0<quiet<.1 and .7<a.mean<1


def test_only_public_counts_and_participants_are_inputs():
    a=TradeActivity();b=TradeActivity()
    for turn in range(20):
        event=dict(turn=turn,actor=turn%4,hand_sizes=(3,5,2,6),trade_participants=[(turn%4,(turn+1)%4)] if turn%3 else [])
        a.observe(**event);b.observe(**event)
    assert a==b


def test_adaptive_endpoint_profiles_and_public_signal_only():
    import run_adaptive as r
    a=r.adaptive_values('A',0);b=r.adaptive_values('A',1)
    assert a==r.PROFILES['N'] and b==r.PROFILES['T']
    assert r.adaptive_values('P',0)==r.PROFILES['N']
    assert r.adaptive_values('P',1)==r.PROFILES['H']
    r.REGIME='none';quiet=r.adaptive_values('P',.3)
    r.REGIME='full';active=r.adaptive_values('P',.3)
    assert quiet==active
