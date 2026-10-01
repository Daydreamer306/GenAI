"""真实课程工具/MCP/TF-IDF + 模拟模型的完整多 Agent 验证。"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from ie_agent.agent import AgentOrchestrator, PromptBuilder
from ie_agent.agent.answer_contract import answers_equal, parse_final_answer
from ie_agent.agent.reviewer import ReviewerAgent, evidence_gate
from ie_agent.agent.solver import SolverAgent
from ie_agent.cli import app
from ie_agent.config import Settings
from ie_agent.contracts import AgentRequest, ModelResult, PromptContext, PromptSource, ReviewVerdict
from ie_agent.knowledge import KnowledgeService
from ie_agent.mas_evaluation import run_comparison
from ie_agent.model.demo import DemoModel
from ie_agent.run_artifacts import RunArtifactWriter
from ie_agent.web.api import create_app


class FixedModel:
    def __init__(self, result):
        self.result = result

    def generate(self, messages):
        return self.result


class MASWorkflowTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        network = patch(
            "httpx.HTTPTransport.handle_request", side_effect=AssertionError("禁止真实外部网络")
        )
        network.start()
        self.addCleanup(network.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(_env_file=None, rag_backend="tfidf", runs_dir=Path(self.temp.name))
        self.query = json.dumps(
            {"tool_name": "entropy", "arguments": {"probabilities": [0.5, 0.5]}}
        )

    def run_agent(self, scenario="repair", **options):
        model = DemoModel(scenario)
        agent = AgentOrchestrator(self.settings, model=model)
        request = AgentRequest(
            session_id="test",
            turn_id="test",
            query=self.query,
            planner="rules",
            structured_answer=True,
            **options,
        )
        return agent.answer(request)

    def test_rejection_feedback_reaches_solver_then_passes(self):
        result = self.run_agent()
        self.assertEqual((result.outcome, result.stop_reason), ("passed", "review_approved"))
        self.assertEqual(len(result.rounds), 2)
        self.assertFalse(result.rounds[0].verdict.approved)
        self.assertTrue(result.rounds[1].verdict.approved)
        self.assertEqual(result.messages[1].recipient, "solver")
        self.assertEqual(result.metrics.model_calls, 4)
        self.assertEqual(result.metrics.execution_mode, "simulated")
        self.assertEqual(result.tool_results[0].result["entropy"], 1)
        self.assertTrue(
            answers_equal(parse_final_answer(result.answer), {"value": 1, "unit": "bit"})
        )
        self.assertEqual([s.step_id for s in result.trace], list(range(1, len(result.trace) + 1)))

    def test_pass_stops_at_one_round(self):
        result = self.run_agent("pass")
        self.assertEqual(len(result.rounds), 1)
        self.assertEqual(result.outcome, "passed")

    def test_rejection_is_bounded_and_draft_is_not_accepted(self):
        result = self.run_agent("reject", max_rounds=3)
        self.assertEqual((result.outcome, result.stop_reason), ("failed", "max_rounds_exceeded"))
        self.assertEqual(len(result.rounds), 3)
        self.assertIn("未通过审核", result.answer)

    def test_solver_failure_exits_without_reviewer(self):
        result = self.run_agent("solver-error")
        self.assertEqual(result.stop_reason, "solver_error")
        self.assertEqual(result.metrics.model_calls, 1)
        self.assertIsNone(result.rounds[0].reviewer_result)

    def test_reviewer_failure_and_invalid_json_never_pass(self):
        for scenario in ("reviewer-error", "invalid-review"):
            with self.subTest(scenario=scenario):
                result = self.run_agent(scenario)
                self.assertEqual(result.outcome, "failed")
                self.assertEqual(result.stop_reason, "reviewer_error")
                self.assertEqual(len(result.rounds), 1)

    def test_tool_validation_failure_stops_before_model(self):
        self.query = json.dumps(
            {"tool_name": "entropy", "arguments": {"probabilities": [0.5, 0.6]}}
        )
        result = self.run_agent()
        self.assertEqual(result.stop_reason, "tool_error")
        self.assertEqual(result.metrics.model_calls, 0)
        self.assertEqual(result.rounds, [])

    def test_single_pass_is_explicitly_unreviewed(self):
        result = self.run_agent(enable_review=False)
        self.assertEqual(result.outcome, "unreviewed")
        self.assertEqual(result.stop_reason, "review_disabled")
        self.assertEqual(result.metrics.model_calls, 1)

    def test_all_submission_artifacts_are_written_and_consistent(self):
        result = self.run_agent()
        path = Path(result.artifact_dir)
        files = {
            "trace.jsonl",
            "report.md",
            "review.json",
            "metrics.json",
            "response.json",
            "request.json",
            "evidence.json",
            "messages.json",
        }
        self.assertTrue(files.issubset({p.name for p in path.iterdir()}))
        trace = [json.loads(line) for line in (path / "trace.jsonl").read_text().splitlines()]
        self.assertEqual(len(trace), len(result.trace))
        self.assertEqual(json.loads((path / "review.json").read_text())["outcome"], "passed")
        self.assertEqual(json.loads((path / "response.json").read_text())["run_id"], result.run_id)

    def test_real_tfidf_evidence_is_used_by_both_roles(self):
        settings = self.settings.model_copy(
            update={"knowledge_dir": Path(self.temp.name) / "knowledge"}
        )
        knowledge = KnowledgeService(settings)
        knowledge.ingest(Path(__file__).resolve().parents[1] / "examples" / "knowledge")
        agent = AgentOrchestrator(settings, model=DemoModel("pass"), knowledge=knowledge)
        result = agent.answer(
            AgentRequest(session_id="test", turn_id="rag", query="什么是信息熵？", planner="rules")
        )
        self.assertEqual(result.outcome, "passed")
        self.assertGreater(len(result.citations), 0)
        self.assertIn("[资料1]", result.rounds[0].solver_result.text)
        evidence = json.loads((Path(result.artifact_dir) / "evidence.json").read_text())
        self.assertGreater(len(evidence["sources"]), 0)

    def test_gate_overrides_fabricated_citation_even_if_reviewer_approves(self):
        context = PromptContext(query="问题", sources=[PromptSource(title="教材", text="真实内容")])
        verdict = ReviewVerdict(approved=True, summary="通过", issues=[])
        for answer in ("引用不存在 [资料2]", "没有引用"):
            with self.subTest(answer=answer):
                self.assertFalse(evidence_gate(context, answer, verdict).approved)

    def test_review_schema_rejects_strings_and_inconsistent_flags(self):
        context = PromptContext(query="问题")
        for payload in [
            {"approved": "true", "summary": "通过", "issues": []},
            {"approved": True, "summary": "通过", "issues": ["错误"]},
            {"approved": False, "summary": "驳回", "issues": []},
        ]:
            reviewer = ReviewerAgent(
                FixedModel(
                    ModelResult(
                        text=json.dumps(payload), provider="mock", model="test", status="success"
                    )
                )
            )
            result, verdict = reviewer.review(context, "草稿")
            self.assertIsNone(verdict)
            self.assertEqual(result.error_type, "invalid_review")

    def test_gate_does_not_accept_wrong_value_based_on_yes_vote(self):
        from ie_agent.contracts import PromptToolEvidence

        context = PromptContext(
            query="熵",
            skill_instruction="<final_answer>",
            tool_evidence=[
                PromptToolEvidence(tool_name="entropy", result={"entropy": 1, "unit": "bit"}),
            ],
        )
        verdict = ReviewVerdict(approved=True, summary="通过", issues=[])
        gated = evidence_gate(
            context, '<final_answer>{"value":2,"unit":"bit"}</final_answer>', verdict
        )
        self.assertFalse(gated.approved)

    def test_solver_includes_rejection_and_previous_draft_in_next_request(self):
        seen = []

        class RecordingModel:
            def generate(self, messages):
                seen.append(messages)
                return ModelResult(text="修订答案", model="test", status="success")

        solver = SolverAgent(RecordingModel(), PromptBuilder())
        feedback = ReviewVerdict(approved=False, summary="错误", issues=["修正单位"])
        solver.solve(PromptContext(query="问题"), "上一轮草稿", feedback)
        self.assertEqual(seen[0][-2].content, "上一轮草稿")
        self.assertIn("修正单位", seen[0][-1].content)

    def test_artifact_writer_redacts_configured_secret(self):
        result = self.run_agent()
        fake_secret = "fake-test-secret-not-a-real-key"
        settings = self.settings.model_copy(
            update={
                "minimax_api_key": Settings(
                    _env_file=None, minimax_api_key=fake_secret
                ).minimax_api_key
            }
        )
        result.answer += fake_secret
        RunArtifactWriter(settings).save(
            AgentRequest(session_id="test", turn_id="secret", query="问题"),
            result,
            PromptContext(query="问题"),
        )
        for path in Path(result.artifact_dir).iterdir():
            self.assertNotIn(fake_secret, path.read_text())

    def test_artifact_failure_returns_failed_state(self):
        with patch.object(RunArtifactWriter, "save", side_effect=OSError("磁盘满")):
            result = self.run_agent()
        self.assertEqual(result.stop_reason, "artifact_write_error")
        self.assertEqual(result.outcome, "failed")

    def test_comparison_uses_same_tasks_and_separate_designs(self):
        path = run_comparison(self.settings, Path("evaluation/mas_tasks.json"), offline=True)
        report = json.loads((path / "comparison.json").read_text())
        self.assertEqual(report["execution_mode"], "simulated")
        self.assertEqual(len(report["items"]), 12)
        summaries = {item["design"]: item for item in report["summary"]}
        self.assertEqual(summaries["single_pass"]["accuracy"], 0)
        self.assertEqual(summaries["review_loop"]["accuracy"], 1)
        self.assertEqual(summaries["review_loop"]["model_calls"], 24)

    def test_web_chat_returns_review_trace_and_saved_artifacts(self):
        api = create_app(
            self.settings, orchestrator=AgentOrchestrator(self.settings, model=DemoModel("repair"))
        )
        with TestClient(api) as client:
            response = client.post("/api/chat", json={"session_id": "webtest", "query": self.query})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["outcome"], "passed")
            self.assertEqual(len(data["rounds"]), 2)
            self.assertTrue(Path(data["artifact_dir"]).is_dir())
            self.assertEqual(client.get("/").status_code, 200)

    def test_failed_cli_demo_exits_nonzero(self):
        with patch("ie_agent.cli.Settings", return_value=self.settings):
            result = CliRunner().invoke(app, ["demo", "--scenario", "reject"])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("max_rounds_exceeded", result.stdout)

    def test_live_benchmark_requires_minimax_key(self):
        with patch("ie_agent.cli.Settings", return_value=self.settings):
            result = CliRunner().invoke(app, ["benchmark", "--limit", "1"])
        self.assertEqual(result.exit_code, 1)

    def test_answer_checker_rejects_boolean_nan_and_extra_answer_tags(self):
        self.assertFalse(answers_equal({"value": True, "unit": "bit"}, {"value": 1, "unit": "bit"}))
        self.assertFalse(
            answers_equal({"value": float("nan"), "unit": ""}, {"value": 0, "unit": ""})
        )
        with self.assertRaises(ValueError):
            parse_final_answer('<final_answer>{"value":1,"unit":""}</final_answer>' * 2)


if __name__ == "__main__":
    unittest.main()
