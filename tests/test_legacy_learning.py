import random
import numpy as np
import pytest
import torch
from agents.minimax_q_agent import MinimaxQAgent
from agents.dqn_agent import DQNAgent,DQNConfig,DQNType
from agents.dyna_q_agent import ProbabilisticTransitionModel
from agents.nash_q_agent import compute_nash_equilibrium_nashpy,NashType


def test_legacy_runner_policy_uses_own_action_rows():
    a=MinimaxQAgent(size=2,alpha=1,seed=4)
    for opp in range(5):a.update(0,3,opp,10,1,True)
    assert all(a.act_col(0,0)==3 for _ in range(20))


def test_legacy_minimax_sampling_is_local_to_agent_seed():
    a=MinimaxQAgent(size=2,seed=4);b=MinimaxQAgent(size=2,seed=4)
    a.Q[0]=np.eye(5);b.Q[0]=np.eye(5)
    left=[];right=[]
    for i in range(30):
        left.append(a.act_row(0,0));np.random.seed(i);np.random.random(50)
        right.append(b.act_row(0,0))
    assert left==right


@pytest.mark.parametrize('done,expected',[(True,1.),(False,2.8)])
def test_legacy_minimax_terminal_update(done,expected):
    a=MinimaxQAgent(size=2,alpha=.5,gamma=.9);a.Q[1,:,:]=4
    a.update(0,2,4,2,1,done)
    assert a.Q[0,2,4]==pytest.approx(expected) and a.dirty[0]


def test_legacy_probabilistic_model_preserves_joint_outcomes():
    m=ProbabilisticTransitionModel();m.update(1,0,1,2,10.,True);m.update(1,0,1,2,-2.,False)
    rng=random.Random(3)
    assert {m.predict(1,0,1,rng) for _ in range(100)}=={(2,10.,True),(2,-2.,False)}


def test_legacy_dqn_warmup_requires_full_batch():
    a=DQNAgent(DQNConfig(state_dim=4,batch_size=4,min_buffer_size=1,hidden_dims=[8]),seed=1)
    for _ in range(3):a.store_transition(np.zeros(4),0,1,np.zeros(4),True)
    assert a.train_step() is None
    a.store_transition(np.zeros(4),0,1,np.zeros(4),True)
    assert np.isfinite(a.train_step())


def test_legacy_dqn_checkpoint_load(tmp_path):
    cfg=DQNConfig(state_dim=4,batch_size=2,min_buffer_size=2,hidden_dims=[8])
    a=DQNAgent(cfg,seed=5)
    for _ in range(4):a.store_transition(np.zeros(4),0,2,np.ones(4),True)
    a.train_step();path=tmp_path/'legacy.pt';a.save(path)
    b=DQNAgent(cfg,seed=2);b.load(path)
    for k in a.q_network.state_dict():torch.testing.assert_close(a.q_network.state_dict()[k],b.q_network.state_dict()[k],rtol=0,atol=0)
    assert a.steps_done==b.steps_done


def test_nash_solution_has_no_profitable_unilateral_deviation():
    q=np.array([[1.,-1.],[-1.,1.]])
    result=compute_nash_equilibrium_nashpy(q,-q,method=NashType.SUPPORT_ENUMERATION)
    assert result.converged
    assert result.pi_a.sum()==pytest.approx(1) and result.pi_b.sum()==pytest.approx(1)
    assert np.max(q@result.pi_b)<=result.value_a+1e-8
    assert np.max(result.pi_a@(-q))<=result.value_b+1e-8


def test_role_conditioned_shared_dqn_keeps_all_config_fields():
    from agents.dqn_agent import MultiAgentDQN
    cfg=DQNConfig(state_dim=4,hidden_dims=[8],min_buffer_size=2,tau=.25,batch_size=2)
    shared=MultiAgentDQN(4,mode='centralized',config=cfg,seed=3)
    assert shared.agent.config.min_buffer_size==2 and shared.agent.config.tau==.25
    s=np.array([0.,.1,.2,.3],dtype=np.float32)
    shared.store_transition_a(s,1,3,s,False)
    shared.store_transition_b(s,2,-3,s,True)
    transitions=list(shared.agent.replay_buffer.buffer)
    np.testing.assert_array_equal(transitions[0].state[:-1],s)
    np.testing.assert_array_equal(transitions[1].state[:-1],s)
    assert [t.state[-1] for t in transitions]==[0,1]


@pytest.mark.integration
def test_wumpus_shared_dqn_training_conditions_on_asymmetric_roles(monkeypatch):
    from train import dqn_training as training
    from env.wumpus_env_extended import WumpusChaseEnvExtended
    from env.maps_extended import small_layout
    real=training.MultiAgentDQN;seen=[]
    def checked(*args,**kwargs):
        seen.append(kwargs['mode'])
        assert kwargs['mode']=='centralized'
        return real(*args,**kwargs)
    monkeypatch.setattr(training,'MultiAgentDQN',checked)
    logs=training.train_dqn_selfplay(WumpusChaseEnvExtended(layout=small_layout(),t_max=2),
             episodes=2,eval_every=1,eval_episodes=1,verbose=False)
    assert seen==['centralized'] and len(logs['win_rate'])==2
