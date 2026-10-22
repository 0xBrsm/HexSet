"""Complete-input grouping, candidate ownership and output restoration."""
import numpy as np

from hexset.clients.batching import coalesce


def feed():
    return dict(observation=np.asarray([[1.,2.],[3.,4.],[1.,2.],[3.,4.]],np.float32),
        flat=np.tile([True,False],(4,1)),setup=np.ones((4,3),bool),
        road_building=np.array([[5,6],[5,6],[9,10],[9,10],[7,8],[7,8]],np.int64),
        road_building_row=np.array([2,0,3,1,2,0],np.int64),
        discard=np.tile(np.arange(10),(4,1)),discard_row=np.arange(4,dtype=np.int64),
        double_rows=np.array([1,3],np.int64))


def test_compound_blocks_rebase_and_scatter_in_the_original_candidate_order():
    f=feed(); before={k:v.copy() for k,v in f.items()}
    g=coalesce(f)
    assert g is not None
    np.testing.assert_array_equal(g.inverse,[0,1,0,1])
    np.testing.assert_array_equal(g.feed['road_building_row'],[0,0,1])
    np.testing.assert_array_equal(g.feed['double_rows'],[1])
    for field in ('road_building','discard'):
        np.testing.assert_array_equal(g.feed[field][g.candidates[field]],f[field])
    np.testing.assert_array_equal(g.feed['double_rows'][g.candidates['double_rows']],g.inverse[f['double_rows']])
    for k,v in f.items():np.testing.assert_array_equal(v,before[k])


def test_a_repeated_row_retains_two_representatives_with_valid_compound_owners():
    f=feed()
    for field in ('observation','flat','setup'): f[field]=np.repeat(f[field][:1],4,axis=0)
    f['road_building']=np.tile([[5,6]],(4,1));f['road_building_row']=np.arange(4,dtype=np.int64)
    f['double_rows']=np.arange(4,dtype=np.int64)
    g=coalesce(f)
    assert g is not None and len(g.feed['observation'])==2
    np.testing.assert_array_equal(g.feed['road_building_row'],[0,1])
    np.testing.assert_array_equal(g.feed['double_rows'],[0,1])
    np.testing.assert_array_equal(g.feed['discard'][g.candidates['discard']],f['discard'])


def test_masks_private_values_and_candidate_order_distinguish_rows():
    for field in ('flat','setup','observation'):
        f=feed(); f[field][2,0]=not f[field][2,0] if field!='observation' else 1.000001
        g=coalesce(f)
        assert g is not None and len(g.feed['observation'])==3
    f=feed(); f['road_building'][[0,4]]=f['road_building'][[4,0]]
    g=coalesce(f)
    assert g is not None and len(g.feed['observation'])==3


def test_small_unique_or_unsupported_batches_keep_their_original_contract():
    assert coalesce({'observation':np.zeros((2,5))}) is None
    assert coalesce({'observation':np.arange(12).reshape(4,3)}) is None
    assert coalesce({'observation':np.zeros((4,3)),'custom':np.zeros(4)}) is None
    f=feed();del f['road_building_row']
    assert coalesce(f) is None


def test_onnx_restores_rows_and_uses_the_current_session_without_cross_call_state():
    import pytest
    pytest.importorskip('onnxruntime')
    from hexset.clients.onnxbot import DecisionPolicy

    class Session:
        factor=1.
        def run(self,outputs,inputs):
            self.count=len(inputs['observation'])
            return [inputs['observation'][:,:1]*self.factor for _ in outputs]
    p=DecisionPolicy.__new__(DecisionPolicy)
    p.session=Session();p._cpu_inference=True;p._dynamic_rows=True;p._fixed_sizes={}
    f=feed()
    for outputs in (['decision'],['value']):
        got=p._run(f,outputs)[outputs[0]]
        assert p.session.count==(2 if outputs==['value'] else 4)
        np.testing.assert_array_equal(got,f['observation'][:,:1])
    p.session.factor=2.
    np.testing.assert_array_equal(p._run(f,['value'])['value'],f['observation'][:,:1]*2)
    p._dynamic_rows=False
    p._run(f,['value']);assert p.session.count==4
    p._dynamic_rows=True;p._fixed_sizes={'road_building':6}
    p._run(f,['value']);assert p.session.count==4
    p._fixed_sizes={};p._cpu_inference=False
    p._run(f,['value']);assert p.session.count==4
