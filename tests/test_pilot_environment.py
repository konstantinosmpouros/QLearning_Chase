from dataclasses import replace
import numpy as np
import pytest


def world(m, mode='outcome', horizon=5):
    return m.World(m.Layout(4, frozenset({(1, 1)}), frozenset({(2, 1)}), (3, 0), (3, 3)),
                   horizon=horizon, reward_mode=mode)


@pytest.mark.parametrize('pos,action,failed,expected', [
    ((0,0),0,False,(0,0)), ((0,0),1,False,(1,0)), ((1,0),2,False,(0,0)),
    ((0,1),3,False,(0,0)), ((0,0),4,False,(0,1)),
    ((0,0),2,False,(0,0)), ((0,0),3,False,(0,0)),
    ((3,2),1,False,(3,2)), ((0,3),4,False,(0,3)),
    ((1,0),4,False,(1,0)), ((0,0),1,True,(0,0)),
])
def test_movement_contract(pilot, pos, action, failed, expected):
    assert world(pilot).move(pos, action, failed) == expected


@pytest.mark.parametrize('a,b,aa,ab,left,winner,reason', [
    ((0,0),(0,2),4,3,5,0,'capture'),
    ((0,0),(0,1),4,3,5,0,'capture'),
    ((3,2),(2,3),4,1,1,0,'capture'), # capture on treasure beats timeout
    ((2,0),(1,1),4,1,5,0,'capture'), # capture on pit still wins
    ((3,2),(0,0),4,0,5,0,'treasure'),
    ((0,0),(3,2),0,4,5,1,'treasure'),
    ((2,0),(0,3),4,0,5,1,'pit'),
    ((0,3),(2,0),0,4,5,0,'pit'),
    ((2,0),(0,3),1,0,5,1,'wumpus'),
    ((0,3),(2,0),0,1,5,0,'wumpus'),
    ((3,2),(2,0),4,4,1,0,'treasure'), # treasure before hazard/timeout
    ((2,0),(3,2),4,4,1,1,'treasure'),
    ((2,0),(2,2),1,3,5,0,'pit'), # both die at different hazards: B death decides
    ((0,0),(0,3),0,0,1,1,'timeout'),
])
def test_terminal_priority_and_rewards(pilot, a,b,aa,ab,left,winner,reason):
    s=pilot.State(a,b,left,0.0)
    ns,r,done,info=world(pilot).transition(s,aa,ab)
    assert done and info['winner']==winner and info['reason']==reason
    np.testing.assert_array_equal(r, [10,-10] if winner==0 else [-10,10])
    assert ns.remaining==left-1


def test_ongoing_and_timeout_are_different(pilot):
    w=world(pilot);s=pilot.State((0,0),(0,3),2,0.)
    ns,r,done,info=w.transition(s,0,0)
    assert not done and info['winner'] is None
    np.testing.assert_array_equal(r,[0,0])
    assert w.transition(ns,0,0)[3]['reason']=='timeout'


@pytest.mark.parametrize('p',[0.,.1,1.])
def test_safe_distinct_reset_and_reproducibility(pilot,p):
    w=world(pilot);r1=np.random.default_rng(7);r2=np.random.default_rng(7)
    for _ in range(40):
        a=w.initial(p,r1);b=w.initial(p,r2)
        assert a==b and a.a!=a.b and a.a in w.safe and a.b in w.safe
        assert a.remaining==5 and a.p_fail==p


def test_failure_endpoints(pilot):
    w=world(pilot);s=pilot.State((0,0),(0,3),5,1.)
    ns,*_=w.step(s,1,1,np.random.default_rng(9));assert (ns.a,ns.b)==(s.a,s.b)
    ns,*_=w.step(replace(s,p_fail=0.),1,1,np.random.default_rng(9))
    assert (ns.a,ns.b)==((1,0),(1,3))


def test_independent_failure_draws(pilot):
    class Controlled:
        def random(self,n):
            assert n==2
            return np.array([.09,.11])
    s=pilot.State((0,0),(0,3),5,.1)
    ns,*_=world(pilot).step(s,1,1,Controlled())
    assert (ns.a,ns.b)==((0,0),(1,3))


def test_potential_telescopes_and_terminal_potential_is_zero(pilot):
    w=world(pilot, 'potential');s=pilot.State((0,0),(0,3),3,.1)
    initial=w.potential(s);total=0.
    for t in range(3):
        ns,r,done,info=w.transition(s,0,0)
        total+=w.gamma**t*info['shaping_a']
        assert r.sum()==pytest.approx(0)
        if done: assert info['shaping_a']==pytest.approx(-w.potential(s))
        s=ns
    assert total==pytest.approx(-initial)


def test_expected_payoff_known_probability(pilot):
    # Catcher alone moves into stationary runner: win iff its action succeeds.
    w=world(pilot);s=pilot.State((0,0),(0,1),5,.25)
    assert w.expected_payoff(s)[4,0]==pytest.approx(7.5)


def test_expected_payoff_cache_cannot_be_corrupted(pilot):
    w=world(pilot);s=pilot.State((0,0),(0,1),5,.25)
    with pytest.raises(ValueError):w.expected_payoff(s)[4,0]=999
    assert w.expected_payoff(s)[4,0]==pytest.approx(7.5)


def test_features_encode_time_noise_and_role_without_flipping_positions(pilot):
    w=world(pilot);s=pilot.State((0,0),(0,3),5,.1)
    a=pilot.features(w,s,0);b=pilot.features(w,s,1)
    np.testing.assert_array_equal(a[:-1],b[:-1]);assert (a[-1],b[-1])==(0,1)
    assert a.shape==(16,) and a.dtype==np.float32
    assert pilot.features(w,replace(s,remaining=1,p_fail=.3),0)[13:15]==pytest.approx([.2,.3])


def test_argmax_does_not_treat_materially_different_large_values_as_ties(pilot):
    class Last:
        def choice(self,items):return items[-1]
    assert pilot.random_argmax([1000.,999.995],Last())==0
