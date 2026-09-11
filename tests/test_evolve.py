from hexset.catanatron.evolve import paired_seed, select_promotions

def test_seed_pairing_and_gate_separation():
    assert [paired_seed("vs-ab2", "discovery", 0, i) for i in range(3)] == [10000000,10000001,10000002]
    assert [paired_seed("vs-shipped", "discovery", 0, i) for i in range(3)] == [10050000,10050001,10050002]
    assert set(range(10000000,10000003)).isdisjoint(range(10050000,10050003))

def test_dual_threshold_promotion():
    rows=[{"candidate":"weak","rows":[{"gate":"vs-ab2","games":100,"wins":{"x":60}},{"gate":"vs-shipped","games":100,"wins":{"x":26}}]}, {"candidate":"strong","rows":[{"gate":"vs-ab2","games":100,"wins":{"x":55}},{"gate":"vs-shipped","games":100,"wins":{"x":30}}]}]
    assert select_promotions(rows,1)==["strong"]

def test_run_resume_fake_pool(tmp_path, monkeypatch):
    import sys, json
    import hexset.catanatron.evolve as e
    calls=[]
    def fake(job):
        calls.append(job)
        cid, gate, stage, gen, idx, seed = job
        return {'candidate':cid,'gate':gate,'stage':stage,'generation':gen,'game_index':idx,'seed':seed,'wins':{'Color.RED':1},'points':{}}
    class Pool:
        def __init__(self, n): self.n=n
        def __enter__(self): return self
        def __exit__(self,*x): pass
        def imap_unordered(self, fn, jobs):
            for j in jobs: yield fake(j)
    monkeypatch.setattr(e, '_play_one', fake)
    monkeypatch.setattr(e, 'Pool', Pool, raising=False)
    ck=tmp_path/'state.json'
    for _ in range(2):
        sys.argv=['evolve','--checkpoint',str(ck),'--count','4','--games','3','--total-games','5','--promote-top','2','--generations','2','--run']
        e.main()
    assert len(calls)==64
    assert len(list((tmp_path/'games').glob('generation-*/*.json')))==64
    state=json.loads(ck.read_text())
    assert [g['generation'] for g in state['generations']] == [0, 1]
    assert all(c['parent_id'].startswith('g00-') for c in state['generations'][1]['candidates'])


def test_dual_gate_uses_candidate_wins_not_opponent():
    rows = [{"candidate": "candidate", "rows": [
        {"gate": "vs-ab2", "games": 100, "wins": {"Color.RED": 20, "Color.BLUE": 80}},
        {"gate": "vs-shipped", "games": 100, "wins": {"Color.RED": 20, "Color.BLUE": 80}},
    ]}]
    assert select_promotions(rows, 1) == []


def test_corrupt_game_checkpoint_refuses_resume(tmp_path, monkeypatch):
    import json, sys
    import hexset.catanatron.evolve as e
    def fake(job):
        cid, gate, stage, gen, idx, seed = job
        return {'candidate': cid, 'gate': gate, 'stage': stage, 'generation': gen,
                'game_index': idx, 'seed': seed, 'wins': {'Color.RED': 1}, 'points': {}}
    class Pool:
        def __init__(self, n): pass
        def __enter__(self): return self
        def __exit__(self, *x): pass
        def imap_unordered(self, fn, jobs):
            for job in jobs: yield fake(job)
    monkeypatch.setattr(e, '_play_one', fake)
    monkeypatch.setattr(e, 'Pool', Pool)
    ck = tmp_path / 'state.json'
    args = ['evolve', '--checkpoint', str(ck), '--count', '1', '--games', '1',
            '--total-games', '1', '--generations', '1', '--run']
    sys.argv = args; e.main()
    game = next((tmp_path / 'games').rglob('*.json'))
    game.write_text('{bad json')
    sys.argv = args
    import pytest
    with pytest.raises(RuntimeError, match='corrupt game checkpoint'):
        e.main()
