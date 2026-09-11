"""Replay evaluated games using the shipped preset, with frozen old opponents.

Run in the pinned campaign image with this checkout at /study and evaluated
788f27e checkout at /evaluated. Existing raw results mount at /out.
"""
import importlib.util
import json
from multiprocessing import get_context
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location('evaluated_runner', '/evaluated/src/hexset/bench/native_search.py')
harness = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = harness
spec.loader.exec_module(harness)
from hexset.arena import PRESETS, spawn, register_entrant_kind

ROOT = Path('/out')
NAME = 'road-zero-exp025'


def factory(entrant, board, rng):
    if entrant.name == NAME:
        bot = spawn(PRESETS['heximax-notrade'], board, rng)
        expected = harness._CONFIG[NAME]
        actual = {k: harness.asdict(bot.evaluator.inner.weights) if k == 'weights'
                  else getattr(bot, k) for k in expected}
        assert actual == expected
        return bot
    return harness._candidate(entrant, board, rng)


def init(config):
    harness.initialize(config)
    register_entrant_kind('native-candidate', factory)


def replay(item):
    gate, index = item
    name = f'{NAME}-{gate}-716000000-{index:05d}.json'
    legacy = ROOT/'holdout-local-attempt1/games'/name
    old_path = legacy if legacy.exists() else ROOT/'holdout-local-attempt1-repaired/games'/name
    old = json.loads(old_path.read_text())
    path = ROOT/'adoption-preflight'/name
    identity = {**old['identity'], 'source_sha256': harness.source_hash(),
                'verification': 'shipped-preset-adoption'}
    harness.play_job(dict(identity=identity, path=str(path)))
    new = json.loads(path.read_text())
    keys = ('winner', 'seat', 'turns', 'points', 'seating', 'actions',
            'action_sha256', 'informed_decisions')
    assert {k: old[k] for k in keys} == {k: new[k] for k in keys}, item
    return dict(gate=gate, index=index, exact_trace_match=True)


def main():
    template = json.loads((ROOT/'holdout-local-attempt1/games'/f'{NAME}-ab2-716000000-00000.json').read_text())['identity']
    config = {NAME: template['phenotype'],
              'frozen-shipped': {**template['frozen_shipped'], 'expansion_value': 0.0}}
    jobs = [(gate, i) for gate in ['ab2', 'shipped'] for i in range(4)] + [('ab2', 1935)]
    with get_context('spawn').Pool(min(30, len(jobs)), initializer=init, initargs=(config,)) as pool:
        matches = pool.map(replay, jobs)
    result = dict(source_sha256=harness.source_hash(), matches=matches,
                  candidate='shipped heximax-notrade preset', frozen_opponents=config['frozen-shipped'])
    harness.atomic(ROOT/'adoption-preflight-verdict.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
