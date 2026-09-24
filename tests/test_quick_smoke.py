import pytest

from aspect.audit import validation_record
from aspect.config import build_overrides


def config(**options):
    return build_overrides(method='RS_Full_FT', workflow='orch_workers', task='math',
                           scale='1.7b', model='/model', output='/run',
                           ray_temp='/tmp/as_q', **options)


def test_quick_profile_requires_explicit_smoke():
    with pytest.raises(ValueError, match='smoke'):
        config(quick_smoke=True)


def test_quick_profile_is_small_and_never_canonical():
    cfg = config(smoke=True, quick_smoke=True, steps=2)
    assert cfg['data.train_batch_size'] == 8
    assert cfg['actor_rollout_ref.actor.ppo_mini_batch_size'] == 8
    assert cfg['actor_rollout_ref.rollout.n'] == 4
    assert cfg['+trainer.inline_full_validation']['expected_rows'] == 8
    assert cfg['+trainer.inline_full_validation']['canonical'] is False
    assert cfg['trainer.save_freq'] == -1
    assert cfg['actor_rollout_ref.actor.optim.lr_warmup_steps'] == 15


def test_formal_profile_is_unchanged():
    cfg = config()
    assert cfg['data.train_batch_size'] == 64
    assert cfg['actor_rollout_ref.rollout.n'] == 8
    assert cfg['+trainer.inline_full_validation']['expected_rows'] == 1412
    assert cfg['+trainer.inline_full_validation']['canonical'] is True


def record():
    return dict(step=2, num_total=8, canonical=False, checkpoint_saved=False,
                per_problem_n_correct=[1, 0] * 4, num_correct=4, accuracy=0.5,
                dataset='dapo_math', split='test', n_rollouts=1, dataset_sha256='a' * 64)


def test_small_online_record_is_accepted_only_as_noncanonical():
    validation_record(record(), 2, canonical=False, expected_rows=8)


def test_small_record_cannot_weaken_formal_validation():
    row = record()
    row['canonical'] = True
    with pytest.raises(ValueError):
        validation_record(row, 2, canonical=True, expected_rows=8)
