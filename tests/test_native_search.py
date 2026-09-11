import json
from pathlib import Path

import pytest

pytest.importorskip('catanatron.game')

from hexset.bench.native_search import FROZEN, initialize, play_job, run


def test_resume_rejects_another_information_model(tmp_path):
    path=tmp_path/'game.json'
    path.write_text(json.dumps({'identity':{'information_model':'memoryless'},'complete':True}))
    with pytest.raises(ValueError, match='incompatible'):
        play_job({'path':str(path),'identity':{'information_model':'native-public-history'}})


def test_run_requires_complete_seating_rotation(tmp_path):
    with pytest.raises(ValueError, match='rotations'):
        run({'control':FROZEN},['control'],['shipped'],1,3,1,tmp_path)


def test_checkpoint_reuse_does_not_start_game(tmp_path):
    path=tmp_path/'game.json';identity={'information_model':'native-public-history'}
    row={'identity':identity,'complete':True,'winner':0}
    path.write_text(json.dumps(row))
    assert play_job({'path':str(path),'identity':identity})==row


def test_frozen_control_matches_shipped_factory():
    import random
    from hexset.arena import Entrant,deal_board,deal_game,entrant_from_name,spawn
    from hexset.actions import apply
    from hexset.bench.native_search import _candidate
    initialize({'control':FROZEN})
    board=deal_board(9001,0);game=deal_game(9001,0,4,board=board)
    control=_candidate(Entrant('control'),board,random.Random(11))
    shipped=spawn(entrant_from_name('heximax-notrade'),board,random.Random(11))
    assert control.evaluator.inner.weights == shipped.evaluator.inner.weights
    for _ in range(24):
        a=control.choose(game);b=shipped.choose(game)
        assert a==b
        apply(game,a)
