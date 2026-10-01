"""离线验证模型适配；所有 key 和响应均为虚构，禁止真实网络。"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from openai import AuthenticationError, RateLimitError

from ie_agent.config import Settings
from ie_agent.contracts import ChatMessage, SkillSpec, ToolSpec
from ie_agent.model import DeepSeekModel, MiniMaxModel, create_model


def response(content="答案", calls=None, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=calls),
                finish_reason=finish_reason,
            )
        ],
        model="MiniMax-M2.7",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
    )


class ModelProviderTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.network = patch("httpx.Client.send", side_effect=AssertionError("网络被禁止"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.settings = Settings(
            _env_file=None,
            minimax_api_key="fake-minimax-key",
            deepseek_api_key="fake-legacy-key",
        )
        self.client = Mock()
        self.client.chat.completions.create.return_value = response()
        self.model = MiniMaxModel(self.settings, client=self.client)
        self.messages = [ChatMessage(role="user", content="测试")]

    def test_default_factory_selects_minimax(self):
        self.assertIsInstance(create_model(self.settings), MiniMaxModel)
        self.assertEqual(self.settings.active_model, "MiniMax-M2.7")

    def test_missing_minimax_key_never_uses_legacy_key(self):
        settings = Settings(_env_file=None, deepseek_api_key="fake-legacy-key")
        model = create_model(settings)
        with patch.object(model, "_create_client") as create:
            result = model.generate(self.messages)
            planned = model.plan("测试", "tool_calls", [], [])
        create.assert_not_called()
        self.assertFalse(settings.has_model_key)
        self.assertEqual(result.error_type, "configuration_error")
        self.assertEqual(result.provider, "minimax")
        self.assertIsNone(planned.route)

    def test_client_uses_minimax_host_and_key(self):
        with patch("ie_agent.model.compatible.OpenAI") as factory:
            self.model._create_client()
        factory.assert_called_once_with(
            api_key="fake-minimax-key",
            base_url="https://api.minimax.cn/v1",
            timeout=120.0,
            max_retries=0,
        )

    def test_generation_omits_deepseek_parameters_and_tracks_provider(self):
        result = self.model.generate(self.messages)
        payload = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(payload["model"], "MiniMax-M2.7")
        self.assertEqual(payload["max_tokens"], 8192)
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("extra_body", payload)
        self.assertEqual(result.provider, "minimax")
        self.assertEqual(result.input_tokens, 10)

    def test_inline_and_unfinished_thinking_are_not_exposed(self):
        for content, expected in [
            ("<think>内部\n思考</think>最终答案", "最终答案"),
            ("<think>未完成的内部思考", ""),
        ]:
            with self.subTest(content=content):
                self.client.chat.completions.create.return_value = response(content)
                result = self.model.generate(self.messages)
                self.assertEqual(result.text, expected)
                self.assertNotIn("思考", result.text)

    def test_truncated_answer_is_not_success(self):
        self.client.chat.completions.create.return_value = response(
            "不完整答案",
            finish_reason="length",
        )
        result = self.model.generate(self.messages)
        self.assertEqual(result.error_type, "truncated_response")

    def test_json_plan_accepts_thinking_and_fences_without_json_mode(self):
        route = {"intent": "knowledge", "need_rag": True, "reason": "需要教材"}
        content = "<think>内部思考</think>```json\n" + json.dumps(route) + "\n```"
        self.client.chat.completions.create.return_value = response(content)
        planned = self.model.plan("测试", "json", [], [])
        self.assertTrue(planned.route.need_rag)
        self.assertNotIn("response_format", self.client.chat.completions.create.call_args.kwargs)
        self.assertEqual(planned.model_result.provider, "minimax")

    def test_native_tool_plan_parses_tools_rag_and_skills(self):
        def call(name, arguments):
            return SimpleNamespace(
                id=name,
                function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
            )

        calls = [
            call("entropy", {"probabilities": [0.5, 0.5]}),
            call("search_course_materials", {"query": "熵", "course_tags": ["information_theory"]}),
            call("activate_skill", {"name": "problem_solving"}),
        ]
        self.client.chat.completions.create.return_value = response(None, calls, "tool_calls")
        planned = self.model.plan(
            "计算信息熵",
            "tool_calls",
            [ToolSpec(name="entropy", description="计算熵", input_schema={"type": "object"})],
            [SkillSpec(name="problem_solving", description="解题")],
        )
        self.assertEqual(planned.route.tool_calls[0].tool_name, "entropy")
        self.assertTrue(planned.route.need_rag)
        self.assertEqual(planned.route.skill_names, ["problem_solving"])
        payload = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(payload["max_tokens"], 8192)
        self.assertNotIn("extra_body", payload)

    def test_invalid_plan_returns_error_for_rule_fallback(self):
        self.client.chat.completions.create.return_value = response("不是 JSON")
        planned = self.model.plan("测试", "json", [], [])
        self.assertIsNone(planned.route)
        self.assertEqual(planned.model_result.status, "error")

    def test_rate_limit_and_auth_errors_do_not_retry_or_fallback(self):
        request = httpx.Request("POST", "https://api.minimax.cn/v1/chat/completions")
        for error_type, status, expected in [
            (AuthenticationError, 401, "authentication_error"),
            (RateLimitError, 429, "rate_limit_error"),
        ]:
            with self.subTest(status=status):
                self.client.chat.completions.create.reset_mock()
                self.client.chat.completions.create.side_effect = error_type(
                    "secret-error-detail",
                    response=httpx.Response(status, request=request),
                    body=None,
                )
                result = self.model.generate(self.messages)
                self.assertEqual(result.error_type, expected)
                self.assertNotIn("secret-error-detail", result.error)
                self.assertEqual(result.provider, "minimax")
                self.client.chat.completions.create.assert_called_once()

    def test_deepseek_remains_explicit_and_preserves_parameters(self):
        settings = self.settings.model_copy(update={"model_provider": "deepseek"})
        self.assertIsInstance(create_model(settings), DeepSeekModel)
        model = DeepSeekModel(settings, client=self.client)
        result = model.generate(self.messages)
        payload = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(payload["model"], "deepseek-v4-pro")
        self.assertEqual(payload["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual(result.provider, "deepseek")


if __name__ == "__main__":
    unittest.main()
