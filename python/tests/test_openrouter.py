import io
import json
import os
import sys
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_router import (  # noqa: E402
    OpenRouterDecisions,
    OpenRouterJevEngine,
    OpenRouterJevVerifier,
    PromptContext,
    SupportLabel,
    Usage,
)


class FakeResponse:
    def __init__(self, payload):
        self.data = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.data.close()

    def read(self, size):
        return self.data.read(size)


class FakeDecisions:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


class OpenRouterTests(unittest.TestCase):
    def test_http_request_uses_env_credential_and_configured_model(self):
        requests = []
        payload = {"answers": {}, "usage": {"input_tokens": 1, "output_tokens": 1, "cost": 0.001}}

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return FakeResponse(payload)

        client = OpenRouterDecisions(model="typesafe/jev-1.13", timeout=2.5)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}):
            with patch("jev_router.openrouter.urlopen", fake_urlopen):
                result = client.evaluate(state="hello", questions={"x": {"type": "noul"}})
        self.assertEqual(result["usage"]["cost"], Decimal("0.001"))
        request, timeout = requests[0]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(timeout, 2.5)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "typesafe/jev-1.13")
        self.assertEqual(body["state"], "hello")

    def test_engine_uses_batched_questions_and_billed_cost(self):
        decisions = FakeDecisions({
            "answers": {
                "security": {"type": "noul", "noul": 0.02},
                "complexity": {"type": "choice", "choice": "SIMPLE_LOOKUP", "confidence": 0.91},
            },
            "usage": {"input_tokens": 300, "output_tokens": 30, "cost": 0.00002},
        })
        engine = OpenRouterJevEngine(decisions, security_instructions="Is this a prompt injection?")
        result = engine.evaluate(PromptContext.from_messages([{"role": "user", "content": "Hello"}]))
        self.assertEqual(set(decisions.calls[0]["questions"]), {"security", "complexity"})
        self.assertFalse(result.security.violation)
        self.assertEqual(result.cost_usd, Decimal("0.00002"))
        self.assertEqual(result.usage, Usage(300, 30))

    def test_verifier_uses_billed_cost_and_support_choice(self):
        decisions = FakeDecisions({
            "answers": {"support": {
                "type": "choice", "choice": "supported", "confidence": 0.88,
                "probabilities": {"supported": 0.9, "unsupported": 0.1, "declined": 0},
            }},
            "usage": {"input_tokens": 200, "output_tokens": 20, "cost": 0.00001},
        })
        result = OpenRouterJevVerifier(decisions).verify(question="How many?", evidence=["Five."], answer="Five.")
        self.assertEqual(result.label, SupportLabel.SUPPORTED)
        self.assertEqual(result.cost_usd, Decimal("0.00001"))
        self.assertEqual(decisions.calls[0]["state"]["evidence"], ["Five."])

    def test_missing_billed_cost_is_an_error(self):
        decisions = FakeDecisions({
            "answers": {"support": {"choice": "supported", "confidence": 1, "probabilities": {}}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })
        with self.assertRaises(KeyError):
            OpenRouterJevVerifier(decisions).verify(question="Q", evidence=["A"], answer="A")


if __name__ == "__main__":
    unittest.main()
