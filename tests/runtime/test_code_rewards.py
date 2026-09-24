import multiprocessing
import os
from concurrent.futures import Future, ProcessPoolExecutor

import pytest

from dashboard.evaluate_checkpoints import (
    _code_executor_init,
    _run_deepcoder_reward_canary,
)
from rllm.rewards.reward_fn import code_reward_fn
from rllm.rewards.reward_types import RewardOutput


class _Executor:
    def __init__(self, result: RewardOutput):
        self.result = result
        self.submission = None

    def submit(self, fn, task, action):
        self.submission = (fn, task, action)
        future = Future()
        future.set_result(self.result)
        return future


def test_deepcoder_canary_uses_a_known_correct_program():
    executor = _Executor(RewardOutput(reward=1.0, is_correct=True))

    _run_deepcoder_reward_canary(executor)

    _, task, action = executor.submission
    assert task["data_source"] == "livecodebench"
    assert task["ground_truth"] == [
        {"input": "17\n", "output": "17\n", "testtype": "stdin"}
    ]
    assert "sys.stdin.read" in action


def test_deepcoder_canary_fails_closed_on_reward_error():
    executor = _Executor(
        RewardOutput(
            reward=0.0,
            is_correct=False,
            metadata={"error_message": "MemoryError"},
        )
    )

    with pytest.raises(RuntimeError, match="MemoryError"):
        _run_deepcoder_reward_canary(executor)


def test_deepcoder_canary_passes_through_spawned_reward_stack():
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=context,
        initializer=_code_executor_init,
    ) as executor:
        _run_deepcoder_reward_canary(executor)


@pytest.mark.skipif(
    os.environ.get("RUN_DEEPCODER_POOL_STRESS") != "1",
    reason="opt-in Linux process-pool stress test",
)
def test_deepcoder_pool_handles_one_full_lane_of_real_reward_work():
    worker_count = int(os.environ.get("DEEPCODER_STRESS_WORKERS", "40"))
    task = {
        "problem": "Echo the input integer.",
        "data_source": "livecodebench",
        "ground_truth": [
            {"input": "17\n", "output": "17\n", "testtype": "stdin"}
        ],
    }
    action = "```python\nimport sys\nprint(sys.stdin.read().strip())\n```"
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_code_executor_init,
    ) as executor:
        futures = [
            executor.submit(code_reward_fn, task, action)
            for _ in range(worker_count)
        ]
        results = [future.result(timeout=120) for future in futures]

    assert all(result.is_correct is True for result in results)
