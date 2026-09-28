import copy
import pickle
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch


def entry(m,family='fp',split='train',role=1,rep=0,checkpoint='final'):
    seed=m.stable_seed(split,role,family,rep)
    if family in ['treasure','ambush']:a=m.Scripted(role,family)
    else:a=m.frozen_copy(m.make_agent(family,role,seed))
    return m.PolicyEntry(f'{split}_{role}_{family}_{rep}_{checkpoint}',split,family,role,seed,checkpoint,a)


def test_holdout_split_and_roles_cannot_leak_to_training(pilot,monkeypatch):
    monkeypatch.setattr(pilot,'TRAIN_FAMILIES',('fp','minimax','dqn'))
    bank=[entry(pilot,f,s,r) for f in ['fp','minimax','dynaq','dqn','treasure','ambush']
          for s in ['train','validation','test'] for r in [0,1]]
    pool=pilot.training_pool(bank,1)
    assert {e.family for e in pool}=={'fp','minimax','dqn'}
    assert all(e.split=='train' and e.role==1 for e in pool)
    ev=pilot.evaluation_entries(bank,1)
    assert all(e.split!='validation' and e.role==1 for e in ev)
    assert pilot.opponent_group(entry(pilot,'dynaq','test'))=='unseen_family'
    assert pilot.opponent_group(entry(pilot,'fp','test'))=='unseen_policy'


def test_sampler_equal_family_weight_not_checkpoint_weight(pilot):
    pool=[entry(pilot,'fp'),entry(pilot,'dqn'),entry(pilot,'dqn',checkpoint='half')]
    sampler=pilot.Sampler(pool,pilot.Condition('D','D',randomized_p=True),pilot.CFG.seeds[0])
    assert len(sampler.cells)==6
    p=sampler.probabilities()
    assert sum(p[i] for i,c in enumerate(sampler.cells) if c[0]=='dqn')==pytest.approx(.5)
    for _ in range(100):
        e,noise=sampler.sample();assert e in pool and noise in pilot.CFG.train_p


def test_fixed_control_never_samples_other_families_or_half_checkpoints(pilot):
    pool=[entry(pilot,'fp'),entry(pilot,'dqn'),entry(pilot,'dqn',checkpoint='half')]
    s=pilot.Sampler(pool,pilot.Condition('A','A','dqn'),pilot.CFG.seeds[0])
    for _ in range(25):
        e,p=s.sample();assert e.family=='dqn' and e.checkpoint=='final' and p==.1


def test_adaptive_sampler_keeps_exploration_floor(pilot):
    s=pilot.Sampler([entry(pilot)],pilot.Condition('E','E',randomized_p=True,adaptive=True),11)
    s.difficulty[:]=[0,.5,1];p=s.probabilities()
    assert p.sum()==pytest.approx(1) and np.all(p>=.2/3) and p[2]>p[1]>p[0]


@pytest.mark.parametrize('family',['fp','minimax','dynaq','dqn'])
def test_frozen_snapshot_is_independent_of_source_learning(pilot,family):
    a=pilot.make_agent(family,1,5);f=pilot.frozen_copy(a);before=pilot.parameter_digest(f)
    s=pilot.State((0,0),(0,3),5,.1)
    for _ in range(40):a.learn(pilot.WORLD,s,1,2,[10.,-10.],s,True)
    assert pilot.parameter_digest(f)==before
    assert pilot.parameter_digest(a)!=before
    if family=='dqn':assert all(not p.requires_grad for p in f.online.parameters())


def test_rollout_is_read_only_and_scenario_reproducible(pilot):
    a=pilot.DQN(0,4);opp=pilot.FP(1)
    before=pickle.dumps(a);counts=copy.deepcopy(opp.counts)
    r1=pilot.rollout(a,opp,0,.3,123)
    r2=pilot.rollout(a,opp,0,.3,123)
    for k in r1:
        if not (isinstance(r1[k],float) and np.isnan(r1[k])):assert r1[k]==r2[k]
    # Tensor pickles contain storage ids: compare state data rather than pickle bytes.
    old=pickle.loads(before)
    assert pilot.parameter_digest(a)==pilot.parameter_digest(old)
    assert a.steps==old.steps and a.updates==old.updates and a.losses==old.losses
    assert a.rng.bit_generator.state==old.rng.bit_generator.state
    assert a.replay.size==old.replay.size==0 and opp.counts==counts
    for k in a.target.state_dict():torch.testing.assert_close(a.target.state_dict()[k],old.target.state_dict()[k],rtol=0,atol=0)
    assert a.optimizer.state_dict()==old.optimizer.state_dict()


def test_adaptive_calibration_uses_only_training_pool_without_learning(pilot,monkeypatch):
    pool=[entry(pilot)]
    sampler=pilot.Sampler(pool,pilot.Condition('E','E',randomized_p=True,adaptive=True),11)
    a=pilot.DQN(0,7);before=pilot.parameter_digest(a);seen=[]
    def fake(agent,opponent,role,p,seed,stream='eval'):
        assert opponent is pool[0].agent and stream=='validation'
        seen.append(p);return {'win':0,'steps':2}
    monkeypatch.setattr(pilot,'rollout',fake)
    spent=sampler.calibrate(a,0,11,10)
    assert spent==2*len(sampler.cells)*pilot.CFG.validation_episodes
    assert len(seen)==len(sampler.cells)*pilot.CFG.validation_episodes
    assert pilot.parameter_digest(a)==before and a.steps==0 and a.replay.size==0


def test_interval_counts_training_seeds_not_evaluation_episodes(pilot):
    frame=pd.DataFrame({'condition':['A']*3,'seed':[11,22,33],'win':[.2,.5,.8]})
    summary=pilot.summarize_seeds(frame,['condition'])
    assert summary.iloc[0].training_seeds==3 and summary.iloc[0]['mean']==pytest.approx(.5)
    single=pilot.seed_interval([.5]);assert np.isnan(single['ci_low'])


@pytest.mark.integration
@pytest.mark.parametrize('role',[0,1])
def test_short_training_save_reload_evaluate_and_resume(pilot,tmp_path,monkeypatch,role):
    # All arms exercised with real DQN updates; two fixed controls for axes/checkpoints.
    bank=[entry(pilot,family=f,role=1-role) for f in ['fp','minimax']]
    bank += [entry(pilot,'ambush','test',1-role)]
    cfg=replace(pilot.CFG,learner_steps=48,eval_episodes=2,test_p=(.1,.3),validation_episodes=1)
    monkeypatch.setattr(pilot,'CFG',cfg)
    conditions=[pilot.Condition('A_fp','A','fp'),pilot.Condition('C_fp','C','fp',True),
                pilot.Condition('B_pool','B'),pilot.Condition('D_pool','D',randomized_p=True),
                pilot.Condition('E_adaptive','E',randomized_p=True,adaptive=True)]
    original={e.identifier:pilot.parameter_digest(e.agent) for e in bank}
    for c in conditions:
        training=pilot.train_condition(bank,role,11,c)
        assert training['complete'] and training['step']==48
        assert training['agent'].updates>0 and np.isfinite(training['agent'].losses).all()
        reloaded=pilot.train_condition(bank,role,11,c)
        assert pilot.parameter_digest(reloaded['agent'])==pilot.parameter_digest(training['agent'])
        before=pilot.parameter_digest(training['agent']);size=training['agent'].replay.size
        result=pilot.evaluate_condition(bank,role,11,c,training)
        assert len(result)==12 and result.win.isin([0,1]).all()
        assert result.steps.between(1,pilot.WORLD.horizon).all()
        assert set(result.group)=={'seen_policy','unseen_family'}
        assert pilot.parameter_digest(training['agent'])==before and training['agent'].replay.size==size
        assert (training['validation_steps']>0)==c.adaptive
    assert original=={e.identifier:pilot.parameter_digest(e.agent) for e in bank}


@pytest.mark.integration
def test_interrupted_training_resumes_exactly(pilot,tmp_path,monkeypatch):
    bank=[entry(pilot)];c=pilot.Condition('resume','D',randomized_p=True)
    monkeypatch.setattr(pilot,'CFG',replace(pilot.CFG,learner_steps=48))
    out=tmp_path/'full';out.mkdir();monkeypatch.setattr(pilot,'OUT',out)
    expected=pilot.train_condition(bank,0,11,c)
    out=tmp_path/'interrupted';out.mkdir();monkeypatch.setattr(pilot,'OUT',out)
    real_save=pilot.atomic_save
    def interrupted(state,path):
        real_save(state,path)
        if isinstance(state,dict) and state.get('step')==24:raise InterruptedError('simulated crash')
    monkeypatch.setattr(pilot,'atomic_save',interrupted)
    with pytest.raises(InterruptedError):pilot.train_condition(bank,0,11,c)
    monkeypatch.setattr(pilot,'atomic_save',real_save)
    resumed=pilot.train_condition(bank,0,11,c)
    assert pilot.parameter_digest(resumed['agent'])==pilot.parameter_digest(expected['agent'])
    assert resumed['logs']==expected['logs'] and resumed['agent'].losses==expected['agent'].losses


def test_evaluation_cache_invalidates_when_learner_changes(pilot,monkeypatch):
    monkeypatch.setattr(pilot,'CFG',replace(pilot.CFG,test_p=(.1,),eval_episodes=1))
    bank=[entry(pilot)];c=pilot.Condition('cache','B')
    a=pilot.DQN(0,1);training={'agent':a,'sampler':pilot.Sampler(bank,c,11)}
    pilot.evaluate_condition(bank,0,11,c,training)
    with torch.no_grad():next(a.online.parameters()).add_(1)
    calls=[]
    def fake(*args,**kwargs):
        calls.append(1)
        return {'win':1,'steps':1,'return':10.,'discounted_return':10.,'reason':'capture','own_hazard':0,'capture_steps':1.}
    monkeypatch.setattr(pilot,'rollout',fake)
    result=pilot.evaluate_condition(bank,0,11,c,training)
    assert calls==[1] and result.iloc[0]['win']==1


def test_evaluation_preserves_nonempty_replay_optimizer_and_target(pilot,monkeypatch):
    cfg=replace(pilot.CFG,batch=2,warmup=2,train_every=1)
    a=pilot.DQN(1,8,cfg);s=pilot.State((0,0),(0,3),5,.1)
    for _ in range(4):a.learn(pilot.WORLD,s,1,3,[1.,-1.],s,True)
    before=copy.deepcopy(a)
    def same(x,y):
        if torch.is_tensor(x):torch.testing.assert_close(x,y,rtol=0,atol=0)
        elif isinstance(x,np.ndarray):np.testing.assert_array_equal(x,y)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:same(x[k],y[k])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for u,v in zip(x,y):same(u,v)
        else:assert x==y
    def forbidden(*args,**kwargs):raise AssertionError('evaluation attempted learning')
    monkeypatch.setattr(a,'learn',forbidden)
    pilot.rollout(a,pilot.Scripted(0,'ambush'),1,.3,881)
    same(a.online.state_dict(),before.online.state_dict())
    same(a.target.state_dict(),before.target.state_dict())
    same(a.optimizer.state_dict(),before.optimizer.state_dict())
    same(vars(a.replay),vars(before.replay))
    same(a.rng.bit_generator.state,before.rng.bit_generator.state)
    assert (a.steps,a.updates,a.losses)==(before.steps,before.updates,before.losses)


def test_valid_cache_reused_but_tampered_csv_recomputed(pilot,monkeypatch):
    monkeypatch.setattr(pilot,'CFG',replace(pilot.CFG,test_p=(.1,),eval_episodes=1))
    bank=[entry(pilot)];c=pilot.Condition('cache_integrity','B')
    tr={'agent':pilot.DQN(0,9),'sampler':pilot.Sampler(bank,c,11)}
    original=pilot.evaluate_condition(bank,0,11,c,tr)
    actual=pilot.rollout;calls=[]
    def counted(*args,**kwargs):calls.append(1);return actual(*args,**kwargs)
    monkeypatch.setattr(pilot,'rollout',counted)
    pilot.evaluate_condition(bank,0,11,c,tr);assert calls==[]
    path=pilot.OUT/'evaluation_0_11_cache_integrity.csv'
    changed=original.copy();changed['win']=99;changed.to_csv(path,index=False)
    result=pilot.evaluate_condition(bank,0,11,c,tr)
    assert calls==[1] and result.win.isin([0,1]).all()


@pytest.mark.integration
def test_bank_build_and_reload_preserves_splits_seeds_and_parameters(pilot,monkeypatch):
    monkeypatch.setattr(pilot,'CFG',replace(pilot.CFG,bank_steps=8))
    bank,cost=pilot.build_bank()
    assert len(cost)==24 and cost.steps.sum()==192
    assert len({e.identifier for e in bank})==len(bank)
    assert len({e.seed for e in bank if e.checkpoint=='final'})==24
    before={e.identifier:pilot.parameter_digest(e.agent) for e in bank}
    def forbidden(*a,**k):raise AssertionError('cache should load saved bank')
    monkeypatch.setattr(pilot,'bootstrap',forbidden)
    loaded,_=pilot.build_bank()
    assert before=={e.identifier:pilot.parameter_digest(e.agent) for e in loaded}
