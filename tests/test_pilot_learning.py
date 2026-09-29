import copy
from dataclasses import replace
import numpy as np
import pytest
import torch
from torch import nn


def test_maximin_mixed_and_pure_games(pilot):
    p,v=pilot.maximin(np.array([[1.,-1.],[-1.,1.]]))
    np.testing.assert_allclose(p,[.5,.5]);assert v==pytest.approx(0)
    p,v=pilot.maximin(np.array([[0.,1.],[3.,2.]]))
    np.testing.assert_allclose(p,[0,1]);assert v==pytest.approx(2)


@pytest.mark.parametrize('role',[0,1])
@pytest.mark.parametrize('done,expected',[(True,1.),(False,2.8)])
def test_minimax_update_uses_own_action_axis_and_terminal_mask(pilot,role,done,expected):
    agent=pilot.Minimax(role,alpha=.5,gamma=.9)
    s=pilot.State((0,0),(0,3),5,.1);ns=replace(s,remaining=4)
    agent.q[agent.key(ns)]=np.full((5,5),4.)
    agent.learn(pilot.WORLD,s,1,3,[2.,2.],ns,done)
    own,opp=(1,3) if role==0 else (3,1)
    assert agent.table(s)[own,opp]==pytest.approx(expected)
    assert np.count_nonzero(agent.table(s))==1
    assert agent.key(s) not in agent.cache


def test_runner_policy_uses_rows_after_runner_update(pilot):
    a=pilot.Minimax(1,alpha=1.);s=pilot.State((0,0),(0,3),5,.1)
    for opponent in range(5):a.learn(pilot.WORLD,s,opponent,3,[0.,7.],s,True)
    p,v=a.policy(s);assert p[3]==pytest.approx(1) and v==pytest.approx(7)


@pytest.mark.parametrize('role',[0,1])
def test_fp_beliefs_are_opponent_actions_and_act_is_read_only(pilot,role):
    a=pilot.FP(role);s=pilot.State((0,0),(0,3),5,.1)
    a.learn(pilot.WORLD,s,1,3,[0,0],s,False)
    expected=np.ones(5);expected[3 if role==0 else 1]+=1
    np.testing.assert_array_equal(a.counts[a.belief_key(s)],expected)
    before=copy.deepcopy(a.counts)
    a.act(pilot.WORLD,s,np.random.default_rng(5))
    for k in before:np.testing.assert_array_equal(before[k],a.counts[k])


def test_fp_runner_best_response_has_correct_sign_and_transpose(pilot):
    class Payoff:
        def expected_payoff(self,s):
            q=np.zeros((5,5));q[:,2]=-5
            return q
    a=pilot.FP(1);s=pilot.State((0,0),(0,3),5,.1)
    assert a.act(Payoff(),s,np.random.default_rng(5))==2


def test_dyna_preserves_joint_outcomes_and_planning_budget(pilot):
    a=pilot.DynaMinimax(1,planning=3);s=pilot.State((0,0),(0,3),5,.1)
    ns=replace(s,remaining=4)
    a.learn(pilot.WORLD,s,1,3,[2.,-2.],ns,False)
    a.learn(pilot.WORLD,s,1,3,[10.,-10.],ns,True)
    outcomes=a.model[(s,3,1)]
    assert outcomes[(ns,-2.,False)]==1 and outcomes[(ns,-10.,True)]==1
    assert len(outcomes)==2 and a.planning_updates==6


def test_dueling_decomposition_and_advantage_shift_invariance(pilot):
    net=pilot.DuelingNetwork(hidden=4)
    with torch.no_grad():
        for p in net.parameters():p.zero_()
        net.value.bias.fill_(3)
        net.advantage.bias.copy_(torch.tensor([0.,1.,2.,3.,4.]))
    x=torch.zeros(2,16);q=net(x)
    torch.testing.assert_close(q,torch.tensor([[1.,2.,3.,4.,5.]]).expand(2,-1))
    with torch.no_grad():net.advantage.bias.add_(100)
    torch.testing.assert_close(net(x),q)


class ConstantQ(nn.Module):
    def __init__(self,values):
        super().__init__();self.values=nn.Parameter(torch.tensor(values,dtype=torch.float32))
    def forward(self,x):return self.values.expand(x.shape[0],-1)


@pytest.mark.parametrize('role',[0,1])
@pytest.mark.parametrize('done',[False,True])
def test_double_dqn_actual_loss_uses_online_selection_target_evaluation(pilot,role,done):
    cfg=replace(pilot.CFG,batch=1,warmup=1,train_every=1,target_every=100,gamma=.5)
    a=pilot.DQN(role,7,cfg)
    a.online=ConstantQ([0,4,1,0,0]);a.target=ConstantQ([100,2,0,0,0])
    a.optimizer=torch.optim.SGD(a.online.parameters(),lr=.01)
    s=pilot.State((0,0),(0,3),5,.1);ns=replace(s,remaining=4)
    # Chaser action 0 has prediction 0; runner action 2 has prediction 1.
    a.learn(pilot.WORLD,s,0,2,[3.,6.],ns,done)
    pred=0 if role==0 else 1;reward=3 if role==0 else 6
    target=reward if done else reward+1 # online chooses 1, target evaluates 2
    expected=abs(pred-target)-.5
    assert a.losses[-1]==pytest.approx(expected)
    assert all(p.grad is None for p in a.target.parameters())
    assert a.replay.a[0]==(0 if role==0 else 2)
    assert a.replay.r[0]==reward


def test_warmup_and_target_sync_schedule(pilot):
    cfg=replace(pilot.CFG,batch=2,warmup=3,train_every=1,target_every=4)
    a=pilot.DQN(0,6,cfg);initial=copy.deepcopy(a.target.state_dict())
    s=pilot.State((0,0),(0,1),5,.1)
    for i in range(1,5):
        a.learn(pilot.WORLD,s,4,0,[10,-10],s,True)
        assert a.updates==max(0,i-2)
        if i<4:
            for k in initial:torch.testing.assert_close(a.target.state_dict()[k],initial[k],rtol=0,atol=0)
    for k,v in a.online.state_dict().items():torch.testing.assert_close(a.target.state_dict()[k],v,rtol=0,atol=0)


def test_replay_wraparound_copies_inputs(pilot):
    b=pilot.Replay(2);x=np.zeros(16,np.float32)
    b.push(x,0,1,x,False);x[:]=9
    assert np.all(b.x[0]==0)
    b.push(x,1,2,x,False);b.push(x,2,3,x,True)
    assert (b.size,b.pos)==(2,1) and set(b.r)=={2,3}
    batch=b.sample(2,np.random.default_rng(0),'cpu')
    assert batch[1].dtype==torch.int64 and batch[0].shape==(2,16)


def test_exact_checkpoint_continuation_includes_optimizer_replay_and_rng(pilot,tmp_path):
    cfg=replace(pilot.CFG,batch=2,warmup=2,train_every=1,target_every=3)
    a=pilot.DQN(1,42,cfg);s=pilot.State((0,0),(0,3),5,.1)
    for i in range(5):a.learn(pilot.WORLD,s,1,i%5,[2.,-2.],s,i%2==0)
    path=tmp_path/'resume.pt';pilot.atomic_save(a,path);b=pilot.trusted_load(path)
    for i in range(7):
        for agent in [a,b]:agent.learn(pilot.WORLD,s,i%5,(i+1)%5,[3.,-3.],s,i%2==1)
    assert pilot.parameter_digest(a)==pilot.parameter_digest(b)
    assert a.losses==b.losses and a.rng.bit_generator.state==b.rng.bit_generator.state
    assert a.steps==b.steps and a.replay.pos==b.replay.pos
    for k in a.target.state_dict():torch.testing.assert_close(a.target.state_dict()[k],b.target.state_dict()[k],rtol=0,atol=0)
