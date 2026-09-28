import pytest
from env.maps_extended import ExtendedMapLayout
from env.wumpus_env_extended import WumpusChaseEnvExtended


def wumpus(a=(0,0),b=(0,3),left=5,p=0.):
    layout=ExtendedMapLayout(4,frozenset({(1,1)}),(3,0),frozenset({(2,1)}),(3,3))
    w=WumpusChaseEnvExtended(layout=layout,p_fail=p,t_max=left,seed=5)
    w.a,w.b=a,b
    return w


@pytest.mark.parametrize('a,b,aa,ab,expected',[
    ((0,0),(0,2),4,3,True),((0,0),(0,1),4,3,True),((0,0),(0,3),0,0,False)])
def test_chase_capture_and_rewards(tag_cls,a,b,aa,ab,expected):
    w=tag_cls(p_fail=0);w.c,w.r=a,b
    _,r,done,info=w.step(aa,ab)
    assert info['capture']==expected and done==expected
    swap = a == w.r and b == w.c
    assert r==pytest.approx((10.01 if swap else 10.03) if expected else -.01)
    assert info['reward_runner']==pytest.approx((-9.99 if swap else -10.01) if expected else 0.)


def test_chase_timeout_and_failure_do_not_fake_capture(tag_cls):
    w=tag_cls(p_fail=1,t_max=1);w.c,w.r=(0,0),(0,3)
    s,r,done,info=w.step(1,1)
    assert s==(0,0,0,3) and done and not info['capture']
    assert info['c_fail'] and info['r_fail']


def test_chase_wall_penalty_has_independent_rewards(tag_cls):
    w=tag_cls(p_fail=0);w.c,w.r=(0,0),(0,3)
    _,r,done,info=w.step(2,0)
    assert not done and r==pytest.approx(-.03) and info['reward_runner']==0
    assert r+info['reward_runner']!=0 # Legacy shaping is not zero-sum.


@pytest.mark.parametrize('a,b,aa,ab,left,winner',[
    ((0,0),(0,2),4,3,5,'A_WIN'),((0,0),(0,1),4,3,5,'A_WIN'),
    ((3,2),(2,3),4,1,1,'A_WIN'),((2,0),(1,1),4,1,5,'A_WIN'),
    ((3,2),(0,0),4,0,5,'A_WIN'),((0,0),(3,2),0,4,5,'B_WIN'),
    ((2,0),(0,3),4,0,5,'B_WIN'),((0,3),(2,0),0,4,5,'A_WIN'),
    ((2,0),(0,3),1,0,5,'B_WIN'),((0,3),(2,0),0,1,5,'A_WIN'),
    ((3,2),(2,0),4,4,1,'A_WIN'),((2,0),(3,2),4,4,1,'B_WIN'),
    ((2,0),(2,2),1,3,5,'A_WIN'),((0,0),(0,3),0,0,1,'B_WIN'),
])
def test_wumpus_priority_and_reward_decomposition(a,b,aa,ab,left,winner):
    w=wumpus(a,b,left);_,r,done,i=w.step(aa,ab)
    assert done and i['outcome']==winner
    assert i['reward_outcome_a']==(10 if winner=='A_WIN' else -10)
    assert i['reward_outcome_b']==-i['reward_outcome_a']
    assert r==pytest.approx(sum(i['reward_'+k+'_a'] for k in ['outcome','step','hazard','bump','perception','chase','treasure']))
    assert i['reward_b']==pytest.approx(sum(i['reward_'+k+'_b'] for k in ['outcome','step','hazard','bump','perception','evade','treasure']))


@pytest.mark.parametrize('p',[0.,1.])
def test_wumpus_movement_failure_and_obstacle(p):
    w=wumpus((1,0),(0,3),p=p);_,_,_,i=w.step(4,1)
    assert w.a==(1,0) and w.b==((1,3) if p==0 else (0,3))
    assert i['a_bumped']==(p==0)


def test_wumpus_reset_and_percepts():
    a,b=wumpus(),wumpus()
    for _ in range(30):
        assert a.reset()==b.reset() and a.a!=a.b
        assert a.a in a._free_cells and a.b in a._free_cells
    assert a.get_perception((2,2)).breeze
    assert a.get_perception((2,0)).stench
    assert a.get_perception((3,3)).glitter
    assert a.get_perception((0,0),bumped=True).bump
