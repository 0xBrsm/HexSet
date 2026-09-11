"""Replay missing original-source games and capture the first adapter failure."""
import json
from pathlib import Path
from hexset.bench.native_search import initialize, play_job
from hexset.catanatron.bot import CatanatronBot
from hexset.actions import legal_actions
from hexset.game import pending_free_roads


def main():
    config=json.loads(Path('/configs/manifest-v10.json').read_text())['candidates']
    initialize(config)
    identity=json.loads(Path('/configs/native-holdout-identity.json').read_text())
    original=CatanatronBot._offer
    current={}
    def inspect(bot,game,mirror):
        try:
            return original(bot,game,mirror)
        except ValueError:
            data=dict(identity=current.copy(),phase=game.phase.name,current_player=game.current_player,
                      free_roads=game.free_roads,pending_free_roads=pending_free_roads(game),
                      owned_roads=game._state.edge_owner.count(game.current_player),
                      native_actions=[repr(a) for a in legal_actions(game)],
                      mirror_actions=[repr(a) for a in mirror.playable_actions],
                      mirror_road_building=mirror.state.is_road_building,
                      mirror_free_roads=mirror.state.free_roads_available,turns=game.turns)
            Path('/out/holdout-adapter-error.json').write_text(json.dumps(data,indent=2))
            print(json.dumps(data),flush=True)
            raise
    CatanatronBot._offer=inspect
    for index in [1935,1961,1970,1980,1981,1982,1983,1984,1985,1986]:
        current.clear();current.update(identity);current.update(index=index,gate='ab2')
        print('Replaying original index',index,flush=True)
        play_job(dict(identity=current.copy(),path=f'/out/diagnose-holdout/{index}.json'))


if __name__=='__main__':
    main()
