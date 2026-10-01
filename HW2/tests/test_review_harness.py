"""不依赖数据库/工具库的协作状态机测试。"""

import unittest

from ie_agent.agent.harness import ReviewHarness
from ie_agent.agent.prompt import PromptBuilder
from ie_agent.agent.reviewer import ReviewerAgent
from ie_agent.agent.solver import SolverAgent
from ie_agent.contracts import ModelResult, PromptContext, PromptToolEvidence
from ie_agent.model.demo import DemoModel


class ReviewHarnessTests(unittest.TestCase):
    def setUp(self):
        self.context = PromptContext(
            query="计算二元等概率熵",
            skill_instruction="<final_answer>",
            tool_evidence=[
                PromptToolEvidence(tool_name="entropy", result={"entropy": 1, "unit": "bit"})
            ],
        )

    def run_harness(self, scenario, max_rounds=3):
        model = DemoModel(scenario)
        return ReviewHarness(
            SolverAgent(model, PromptBuilder()), ReviewerAgent(model), max_rounds
        ).run(self.context)

    def test_repair(self):
        result = self.run_harness("repair")
        self.assertEqual(result.outcome, "passed")
        self.assertEqual(len(result.rounds), 2)
        self.assertFalse(result.rounds[0].verdict.approved)
        self.assertTrue(result.rounds[1].verdict.approved)

    def test_failure_stops(self):
        for scenario, reason, rounds in [
            ("reject", "max_rounds_exceeded", 3),
            ("solver-error", "solver_error", 1),
            ("reviewer-error", "reviewer_error", 1),
            ("invalid-review", "reviewer_error", 1),
        ]:
            with self.subTest(scenario=scenario):
                result = self.run_harness(scenario)
                self.assertEqual(result.outcome, "failed")
                self.assertEqual(result.stop_reason, reason)
                self.assertEqual(len(result.rounds), rounds)

    def test_hard_gate_overrides_yes_vote(self):
        model = DemoModel("reject")

        class YesModel:
            def generate(self, messages):
                return ModelResult(
                    text='{"approved":true,"summary":"通过","issues":[]}',
                    model="test",
                    status="success",
                )

        result = ReviewHarness(
            SolverAgent(model, PromptBuilder()), ReviewerAgent(YesModel()), 2
        ).run(self.context)
        self.assertEqual(result.outcome, "failed")
        self.assertEqual(result.stop_reason, "max_rounds_exceeded")

    def test_empty_or_unexpected_model_error_never_passes(self):
        class BrokenModel:
            def generate(self, messages):
                raise RuntimeError("private-detail")

        result = ReviewHarness(
            SolverAgent(BrokenModel(), PromptBuilder()), ReviewerAgent(DemoModel()), 3
        ).run(self.context)
        self.assertEqual(result.stop_reason, "solver_error")
        self.assertNotIn("private-detail", result.rounds[0].solver_result.error)

    def test_empty_answer_stops(self):
        class EmptyModel:
            def generate(self, messages):
                return ModelResult(text="", model="test", status="success")

        result = ReviewHarness(
            SolverAgent(EmptyModel(), PromptBuilder()), ReviewerAgent(DemoModel()), 3
        ).run(self.context)
        self.assertEqual(result.rounds[0].solver_result.error_type, "empty_answer")


if __name__ == "__main__":
    unittest.main()
