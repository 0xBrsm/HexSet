import json
from pathlib import Path
import pytest
from hexset.catanatron.followon_pipeline import run_pipeline

def write(p, x): Path(p).write_text(json.dumps(x))

def test_pass_stops_without_followon(tmp_path):
    prior=tmp_path/'prior'; write(prior, {'status':'PASS'})
    called=[]
    out=run_pipeline(prior=prior, checkpoint={}, root=tmp_path/'out', source_hash='x', cheap_run=lambda _:called.append(1), validation_run=None, evolve_run=None)
    assert out['status']=='PASS' and not called

def test_rejection_resumes_phases_and_is_resumable(tmp_path):
    prior=tmp_path/'prior'; write(prior, {'status':'HOLDOUT_REJECTED'})
    cp=tmp_path/'cp'; write(cp, {'source_hash':'x'})
    calls=[]
    def cheap(p): calls.append('cheap'); return {'status':'NO_CANDIDATE'}
    def validation(c,o): calls.append('validation'); return {'status':'FAIL'}
    def evolve(c,o): calls.append('evolve'); return {'status':'NO_CANDIDATE'}
    out=run_pipeline(prior=prior, checkpoint=cp, root=tmp_path/'out', source_hash='x', cheap_run=cheap, validation_run=validation, evolve_run=evolve)
    assert out['stage']=='evolve' and calls==['cheap','evolve','validation']
    out2=run_pipeline(prior=prior, checkpoint=cp, root=tmp_path/'out', source_hash='x', cheap_run=lambda _: (_ for _ in ()).throw(AssertionError()), validation_run=validation, evolve_run=evolve)
    assert out2['stage']=='evolve'

def test_rejects_unresolved_input_or_hash_mismatch(tmp_path):
    prior=tmp_path/'prior'; write(prior, {'status':'RUNNING'})
    cp=tmp_path/'cp'; write(cp, {'source_hash':'x'})
    with pytest.raises(ValueError): run_pipeline(prior=prior, checkpoint=cp, root=tmp_path/'o', source_hash='x', cheap_run=lambda _:None, validation_run=None, evolve_run=None)
    write(prior, {'status':'HOLDOUT_REJECTED'})
    with pytest.raises(ValueError): run_pipeline(prior=prior, checkpoint=cp, root=tmp_path/'o', source_hash='y', cheap_run=lambda _:None, validation_run=None, evolve_run=None)

def test_closed_holdout_artifact_derives_rejection(tmp_path):
    prior=tmp_path/'prior'; prior.mkdir(); write(prior/'manifest.json', {'status':'COMPLETE','candidate':'x','source_hash':'x'})
    write(prior/'holdout-decision-4096.json', {'decision':'REJECTED_AT_LOOK','candidate':'x'})
    cp=tmp_path/'cp'; write(cp, {'source_hash':'x'})
    seen=[]
    out=run_pipeline(prior=prior, checkpoint=cp, root=tmp_path/'out', source_hash='x', cheap_run=lambda p: seen.append(1) or {'status':'NO_CANDIDATE'}, validation_run=lambda c,o:{'status':'FAIL'}, evolve_run=lambda c,o:{'status':'NO_CANDIDATE'})
    assert out['stage']=='evolve' and seen==[1]

def test_active_prior_artifact_is_rejected(tmp_path):
    prior=tmp_path/'prior'; prior.mkdir(); write(prior/'manifest.json', {'status':'ACTIVE'})
    with pytest.raises(ValueError): run_pipeline(prior=prior, checkpoint={}, root=tmp_path/'o', source_hash='x', cheap_run=lambda p:None, validation_run=None, evolve_run=None)

def test_main_builds_real_evolve_argv_and_fresh_validation(monkeypatch, tmp_path):
    import hexset.catanatron.followon_pipeline as m
    prior=tmp_path/'prior'; write(prior, {'status':'HOLDOUT_REJECTED'})
    cp=tmp_path/'cp'; write(cp, {'source_hash':'src'})
    calls=[]
    monkeypatch.setattr('hexset.catanatron.cheap_followon.run', lambda root, python='python': {'status':'NO_CANDIDATE'})
    monkeypatch.setattr('hexset.catanatron.evolve_validation.run_fresh', lambda checkpoint, out: calls.append(('fresh',str(checkpoint))) or {'status':'PASS'})
    def fake_run(argv, check=True):
        calls.append(argv)
        Path(argv[argv.index('--checkpoint')+1]).parent.mkdir(parents=True, exist_ok=True)
        write(Path(argv[argv.index('--checkpoint')+1]), {'source_hash':'src','generations':[]})
    monkeypatch.setattr(m.subprocess, 'run', fake_run)
    assert m.main(['--prior',str(prior),'--checkpoint',str(cp),'--root',str(tmp_path/'out'),'--source-hash','src']) == 0
    argv=next(x for x in calls if isinstance(x,list))
    assert all(x in argv for x in ('--run','--generations','6','--count','12','--games','120','--total-games','300','--promote-top','4'))
    assert any(x[0]=='fresh' for x in calls)

def test_real_post_confirmation_directory_handoff_and_source_separation(tmp_path):
    prior=tmp_path/'post'; prior.mkdir()
    write(prior/'manifest.json', {'schema':1,'source_hash':'t1-source-A','status':'COMPLETE'})
    write(prior/'confirmation.json', {'candidate':'win-T1','rows':[{'gate':'vs-ab2','games':1024,'candidate_wins':441},{'gate':'vs-shipped','games':1024,'candidate_wins':269}]})
    cp=tmp_path/'cp'; write(cp, {'source_hash':'followon-source-B'})
    calls=[]
    out=run_pipeline(prior=prior, checkpoint=cp, root=tmp_path/'out', source_hash='followon-source-B', cheap_run=lambda p: calls.append('cheap') or {'status':'NO_CANDIDATE'}, validation_run=lambda c,o: calls.append('validation') or {'status':'FAIL'}, evolve_run=lambda c,o: calls.append('evolve') or {'status':'NO_CANDIDATE'})
    assert out['stage']=='evolve' and calls==['cheap','evolve','validation']

def test_actual_cheap_pass_contract_stops_before_evolution(tmp_path):
    prior=tmp_path/'prior'; write(prior, {'status':'HOLDOUT_REJECTED'})
    cp=tmp_path/'cp'; write(cp, {'source_hash':'src'})
    calls=[]
    cheap={'status':'PASSED_VALIDATION','screen':[{'arm':'bankext','gate':'ab2','games':4096,'wins':2800},{'arm':'bankext','gate':'shipped','games':4096,'wins':1600}], 'gates':{'ab2':{'wins':2800,'games':4096},'shipped':{'wins':1600,'games':4096}}}
    out=run_pipeline(prior=prior, checkpoint=cp, root=tmp_path/'out', source_hash='src', cheap_run=lambda p: cheap, validation_run=lambda *x: calls.append('validation'), evolve_run=lambda *x: calls.append('evolve'))
    assert out['status']=='PASS' and out['stage']=='cheap' and calls==[]

def test_fail_prior_without_complete_matrix_rejected(tmp_path):
    prior=tmp_path/'prior'; write(prior, {'status':'FAIL','reason':'workers stopped'})
    cp=tmp_path/'cp'; write(cp, {'source_hash':'src'})
    with pytest.raises(ValueError): run_pipeline(prior=prior, checkpoint=cp, root=tmp_path/'out', source_hash='src', cheap_run=lambda p:None, validation_run=None, evolve_run=None)


def test_cheap_point_pass_or_short_screen_cannot_pass_strict_wilson(tmp_path):
    prior=tmp_path/'prior'; write(prior, {'status':'HOLDOUT_REJECTED'})
    cp=tmp_path/'cp'; write(cp, {'source_hash':'src'})
    for a,b in ((1638,1229),(2130,1085)):
        cheap={'status':'PASSED_VALIDATION','gates':{'ab2':{'wins':a,'games':4096},'shipped':{'wins':b,'games':4096}}}
        with pytest.raises(ValueError):
            run_pipeline(prior=prior, checkpoint=cp, root=tmp_path/f'out-{a}', source_hash='src', cheap_run=lambda p, cheap=cheap: cheap, validation_run=lambda *x:None, evolve_run=lambda *x:None)
