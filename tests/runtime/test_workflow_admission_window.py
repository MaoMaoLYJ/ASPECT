import asyncio
from types import SimpleNamespace

import pytest

from rllm.agents.agent import Episode
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.workflows.workflow import TerminationReason, Workflow


class _AdmissionProbe:
    def __init__(self):
        self.started: list[str] = []
        self.finished: list[str] = []
        self.first_two_started = asyncio.Event()
        self.release_slow = asyncio.Event()
        self.refill_started = asyncio.Event()
        self.blocked_name: str | None = None


class _ProbeWorkflow(Workflow):
    def __init__(self, probe: _AdmissionProbe, **kwargs):
        super().__init__(**kwargs)
        self.probe = probe

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        del uid, kwargs
        start_index = len(self.probe.started)
        name = task["name"]
        self.probe.started.append(name)

        if len(self.probe.started) == 2:
            self.probe.first_two_started.set()

        if start_index == 0:
            await self.probe.first_two_started.wait()
        elif start_index == 1:
            self.probe.blocked_name = name
            await self.probe.release_slow.wait()
        else:
            self.probe.refill_started.set()

        self.probe.finished.append(name)
        return Episode(termination_reason=TerminationReason.UNKNOWN)


class _CodeStageProbe:
    def __init__(self):
        self.generation_started: list[str] = []
        self.judging_started: list[str] = []
        self.two_generations_started = asyncio.Event()
        self.refill_generation_started = asyncio.Event()
        self.release_judging = asyncio.Event()


class _BlockingCodeScheduler:
    def __init__(self, probe: _CodeStageProbe):
        self.probe = probe

    async def submit(self, task: dict, action: str):
        del action
        self.probe.judging_started.append(task["name"])
        await self.probe.release_judging.wait()
        return object()


class _CodeStageWorkflow(Workflow):
    def __init__(self, probe: _CodeStageProbe, **kwargs):
        super().__init__(**kwargs)
        self.probe = probe

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        del uid, kwargs
        self.probe.generation_started.append(task["name"])
        if len(self.probe.generation_started) == 2:
            self.probe.two_generations_started.set()
        if len(self.probe.generation_started) > 2:
            self.probe.refill_generation_started.set()
        await self.run_in_code_executor(None, task, "code")
        return Episode(termination_reason=TerminationReason.UNKNOWN)


async def _run_rolling_refill_case():
    probe = _AdmissionProbe()
    engine = AgentWorkflowEngine(
        workflow_cls=_ProbeWorkflow,
        workflow_args={"probe": probe},
        rollout_engine=None,
        n_parallel_tasks=2,
    )
    engine._rollout_log_interval = 0

    runner = asyncio.create_task(
        engine.execute_tasks(
            [{"name": "first"}, {"name": "slow"}, {"name": "refill"}],
            task_ids=["first", "slow", "refill"],
        )
    )
    try:
        await asyncio.wait_for(probe.first_two_started.wait(), timeout=1)
        await asyncio.wait_for(probe.refill_started.wait(), timeout=1)

        assert probe.release_slow.is_set() is False
        assert len(probe.started) == 3
        assert probe.blocked_name not in probe.finished
    finally:
        probe.release_slow.set()

    results = await asyncio.wait_for(runner, timeout=1)
    engine.executor.shutdown(wait=True)
    return engine, results


def test_workflow_window_refills_before_the_slowest_reward_finishes(capsys):
    engine, results = asyncio.run(_run_rolling_refill_case())

    assert len(results) == 3
    assert engine._admission_started_rollouts == 3
    assert engine._admission_active_rollouts == 0
    assert engine._admission_peak_active_rollouts == 2

    output = capsys.readouterr().out
    assert "window=2 scheduling=rolling_completion_refill" in output
    assert "[WorkflowAdmission] complete total=3 peak_active=2" in output


async def _run_code_stage_pipeline_case():
    probe = _CodeStageProbe()
    scheduler = _BlockingCodeScheduler(probe)
    engine = AgentWorkflowEngine(
        workflow_cls=_CodeStageWorkflow,
        workflow_args={"probe": probe},
        rollout_engine=None,
        n_parallel_tasks=4,
        generation_admission_window=2,
    )
    engine.batch_test_scheduler = scheduler
    engine._rollout_log_interval = 0

    runner = asyncio.create_task(
        engine.execute_tasks(
            [{"name": str(index)} for index in range(4)],
            task_ids=[str(index) for index in range(4)],
        )
    )
    try:
        await asyncio.wait_for(probe.two_generations_started.wait(), timeout=1)
        await asyncio.wait_for(probe.refill_generation_started.wait(), timeout=1)
        assert probe.release_judging.is_set() is False
        assert len(probe.judging_started) >= 1
        assert len(probe.generation_started) > 2
    finally:
        probe.release_judging.set()

    results = await asyncio.wait_for(runner, timeout=1)
    engine.executor.shutdown(wait=True)
    return engine, results


def test_code_stage_transition_refills_generation_before_judging_finishes(capsys):
    engine, results = asyncio.run(_run_code_stage_pipeline_case())

    assert len(results) == 4
    assert engine._generation_stage_peak == 2
    assert engine._generation_stage_releases == 4
    assert engine._generation_stage_active == 0

    output = capsys.readouterr().out
    assert "generation_window=2 workflow_capacity=4" in output
    assert "stage_releases=4 generation_active=0" in output


def test_engine_fails_closed_when_configured_generation_window_is_not_activated():
    config = SimpleNamespace(
        rllm=SimpleNamespace(
            workflow=SimpleNamespace(generation_admission_window=2)
        )
    )

    with pytest.raises(RuntimeError, match="generation admission window mismatch"):
        AgentWorkflowEngine(
            workflow_cls=_ProbeWorkflow,
            workflow_args={"probe": _AdmissionProbe()},
            rollout_engine=None,
            config=config,
            n_parallel_tasks=4,
        )
