import asyncio

import pytest

from examples.math_reasoning.orchestrator_workers_math_workflow import OrchestratorWorkersMathWorkflow
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.orchestrator_workers_workflow import ProposalResult
from rllm.workflows.paper_trajectory_diagnostics import normalized_first_strategy_label
from rllm.workflows.workflow import TerminationReason


@pytest.mark.parametrize("strategies,expected", [
    ([], ""), ([""], ""), ([" \t\r\n "], ""),
    (["", "Do not substitute the second strategy"], ""),
    (["  Algebra  first\nThen substitution"], "algebra first"),
])
def test_strategy_label_handles_empty_content_without_substitution(strategies, expected):
    before = list(strategies)
    assert normalized_first_strategy_label(strategies) == expected
    assert strategies == before


class FixedRollout(RolloutEngine):
    def __init__(self, proposal):
        self.proposal = proposal
        self.calls = []

    async def get_model_response(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        content = self.proposal if kwargs["agent_name"] == "orchestrator" else r"\boxed{42}"
        return ModelOutput(content=content, text=content, reasoning="", completion_ids=[1, 2])


class TooManyStrategies(OrchestratorWorkersMathWorkflow):
    def parse_proposals(self, response):
        return ProposalResult(strategies=["", "algebra", "geometry", "counting"],
                              label="test", execution_mode="parallel")


@pytest.mark.parametrize("instance_routes", [False, True])
@pytest.mark.parametrize("proposal", ["", " \n ", "STRATEGY 1: Algebra"])
@pytest.mark.parametrize("correct", [False, True])
def test_full_orch_empty_strategy_does_not_retry_or_change_rewards(instance_routes, proposal, correct):
    rollout = FixedRollout(proposal)
    reward_calls = []

    def reward(task, response):
        reward_calls.append(response)
        return RewardOutput(reward=float(correct), is_correct=correct)

    async def run():
        engine = AgentWorkflowEngine(OrchestratorWorkersMathWorkflow,
            {"reward_function": reward, "prompts": {}, "max_subtasks": 3,
             "use_final_outcome_reward": True, "agent_lorasb_instance_routing": instance_routes},
            rollout, n_parallel_tasks=1, retry_limit=3)
        try:
            await engine.initialize_pool()
            _, _, episode = await engine.process_task_with_retry(
                {"question": "What is 6 * 7?", "ground_truth": "42"}, "case", 0)
            return episode
        finally:
            engine.executor.shutdown()

    episode = asyncio.run(run())
    assert episode.termination_reason != TerminationReason.ERROR
    assert "error" not in episode.info
    assert len(rollout.calls) == 3 and len(reward_calls) == 1
    routes = [call[1]["agent_name"] for call in rollout.calls]
    assert routes == ["orchestrator", "worker0" if instance_routes else "worker", "synthesizer"]
    assert episode.is_correct is correct and episode.metrics["success"] == int(correct)
    assert all(t.reward == float(correct) for t in episode.trajectories)
    assert all(s.reward == float(correct) for t in episode.trajectories for s in t.steps)
    expected = "algebra" if proposal.startswith("STRATEGY") else ""
    assert episode.metrics["paper_diag/orchestrator/first_strategy_label"] == expected
    if not expected:
        assert episode.trajectories[0].steps[0].action["strategies"] == [""]


def test_over_limit_negative_episode_keeps_zero_reward_with_empty_strategy():
    rollout = FixedRollout("")
    workflow = TooManyStrategies(rollout, reward_function=lambda *_: pytest.fail("No judging expected"),
                                 prompts={}, max_subtasks=3)
    episode = asyncio.run(workflow.run_with_termination_handling({"question": "x"}, "over-limit"))
    assert episode.termination_reason != TerminationReason.ERROR
    assert len(rollout.calls) == 1
    assert not episode.is_correct and episode.metrics["success"] == 0
    assert len(episode.trajectories) == 1 and episode.trajectories[0].reward == 0
    assert episode.metrics["paper_diag/orchestrator/first_strategy_label"] == ""
