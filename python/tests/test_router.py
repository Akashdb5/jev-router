import json
import sys
import unittest
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_router import (  # noqa: E402
    AnthropicMessagesProvider,
    Complexity,
    ComplexityAssessment,
    Completion,
    GateUnavailable,
    JevAssessment,
    ModelTarget,
    OpenAIChatProvider,
    Price,
    PromptBlocked,
    PromptContext,
    Router,
    RouterPolicy,
    SecurityAssessment,
    TypeSafeJevEngine,
    Usage,
)


class Noul:
    def __init__(self, result):
        self.result = result
        self.seen = None

    def evaluate(self, context):
        self.seen = context
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class Choice:
    def __init__(self, result):
        self.result = result
        self.seen = None

    def classify(self, context):
        self.seen = context
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeProvider:
    def __init__(self, usage=Usage(100, 20)):
        self.calls = []
        self.usage = usage

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return Completion("answer", self.usage, {"provider": "fake"})


def make_router(noul=None, choice=None, **kwargs):
    cheap, costly = FakeProvider(), FakeProvider()
    events = []
    router = Router(
        noul=noul or Noul(SecurityAssessment(False, 0.99)),
        choice=choice or Choice(ComplexityAssessment(Complexity.SIMPLE_LOOKUP, 0.99)),
        economy=ModelTarget("cheap", "economy-model", Price(Decimal("1"), Decimal("2"))),
        frontier=ModelTarget("costly", "frontier-model", Price(Decimal("10"), Decimal("20"))),
        providers={"cheap": cheap, "costly": costly},
        on_event=events.append,
        **kwargs,
    )
    return router, cheap, costly, events


MESSAGES = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Hi"}]


class RouterTests(unittest.TestCase):
    def test_shared_routing_cases(self):
        fixtures = Path(__file__).resolve().parents[2] / "spec" / "routing-cases.json"
        for case in json.loads(fixtures.read_text(encoding="utf-8")):
            with self.subTest(case=case["name"]):
                noul = Noul(SecurityAssessment(case["violation"], case["security_confidence"]))
                choice = Choice(ComplexityAssessment(Complexity(case["complexity"]), case["complexity_confidence"]))
                router, cheap, costly, events = make_router(noul, choice)
                if case["expected"] == "blocked":
                    with self.assertRaises(PromptBlocked):
                        router.create(MESSAGES)
                    self.assertEqual(len(cheap.calls) + len(costly.calls), 0)
                else:
                    result = router.create(MESSAGES)
                    self.assertEqual(result.decision.route, case["expected"])
                    self.assertEqual(len(cheap.calls), int(case["expected"] == "economy"))
                    self.assertEqual(len(costly.calls), int(case["expected"] == "frontier"))
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0].route, case["expected"])
                self.assertIn("system: Be concise.", noul.seen.screening_text)

    def test_noul_failure_never_calls_provider(self):
        router, cheap, costly, _ = make_router(noul=Noul(RuntimeError("unavailable")))
        with self.assertRaises(GateUnavailable):
            router.create(MESSAGES)
        self.assertFalse(cheap.calls or costly.calls)

    def test_choice_failure_routes_frontier(self):
        router, cheap, costly, _ = make_router(choice=Choice(RuntimeError("unavailable")))
        result = router.create(MESSAGES)
        self.assertEqual(result.decision.reason, "choice_unavailable")
        self.assertFalse(cheap.calls)
        self.assertEqual(len(costly.calls), 1)

    def test_batched_jev_block_records_its_cost_without_provider_call(self):
        engine = SimpleNamespace(evaluate=lambda context: JevAssessment(
            SecurityAssessment(True, 0.96),
            ComplexityAssessment(Complexity.SIMPLE_LOOKUP, 0.99),
            Usage(200, 10),
            Decimal("0.00002"),
        ))
        router, cheap, costly, events = make_router()
        router.jev = engine
        with self.assertRaises(PromptBlocked):
            router.create(MESSAGES)
        self.assertFalse(cheap.calls or costly.calls)
        self.assertEqual(events[0].router_cost_usd, Decimal("0.00002"))
        self.assertEqual(events[0].jev_usage, Usage(200, 10))

    def test_cost_is_explicitly_estimated(self):
        router, _, _, events = make_router(router_overhead_usd=Decimal("0.0001"))
        result = router.create(MESSAGES)
        self.assertEqual(result.cost.estimated_frontier_baseline_usd, Decimal("0.0014"))
        self.assertEqual(result.cost.actual_provider_usd, Decimal("0.00014"))
        self.assertEqual(result.cost.estimated_savings_usd, Decimal("0.00116"))
        self.assertEqual(events[0].cost, result.cost)
        self.assertFalse(hasattr(events[0], "messages"))

    def test_missing_usage_has_no_cost(self):
        router, cheap, _, events = make_router()
        cheap.usage = None
        result = router.create(MESSAGES)
        self.assertIsNone(result.cost)
        self.assertIsNone(events[0].cost)

    def test_input_validation(self):
        router, _, _, _ = make_router()
        with self.assertRaises(ValueError):
            router.create([{"role": "user", "content": [{"type": "image_url"}]}])
        with self.assertRaises(ValueError):
            router.create(MESSAGES, max_tokens=0)
        with self.assertRaises(ValueError):
            RouterPolicy(economy_threshold=0.5, low_confidence_threshold=0.7)

    def test_metric_callback_failure_does_not_fail_request(self):
        router, _, _, _ = make_router()
        router.on_event = lambda event: (_ for _ in ()).throw(RuntimeError("metrics down"))
        self.assertEqual(router.create(MESSAGES).text, "answer")


class AdapterTests(unittest.TestCase):
    def test_openai_adapter_preserves_messages_and_usage(self):
        calls = []
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: (calls.append(kwargs), response)[1]
        )))
        result = OpenAIChatProvider(client).complete(
            model="test", context=PromptContext.from_messages(MESSAGES), max_tokens=12, temperature=0.2,
        )
        self.assertEqual(calls[0]["messages"], MESSAGES)
        self.assertEqual(calls[0]["temperature"], 0.2)
        self.assertEqual(result.usage, Usage(7, 3))

    def test_anthropic_adapter_hoists_initial_system(self):
        calls = []
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hello")],
            usage=SimpleNamespace(input_tokens=8, output_tokens=4),
        )
        client = SimpleNamespace(messages=SimpleNamespace(
            create=lambda **kwargs: (calls.append(kwargs), response)[1]
        ))
        result = AnthropicMessagesProvider(client).complete(
            model="test", context=PromptContext.from_messages(MESSAGES), max_tokens=12, temperature=None,
        )
        self.assertEqual(calls[0]["system"], "Be concise.")
        self.assertEqual(calls[0]["messages"], [{"role": "user", "content": "Hi"}])
        self.assertNotIn("temperature", calls[0])
        self.assertEqual(result.usage, Usage(8, 4))

    def test_typesafe_jev_batches_questions_and_tracks_usage(self):
        calls = []
        response = SimpleNamespace(
            nouls={"security": SimpleNamespace(noul=0.03)},
            choices={"complexity": SimpleNamespace(choice="SIMPLE_LOOKUP", confidence=0.90)},
            usage=SimpleNamespace(input_tokens=300, output_tokens=10),
        )
        client = SimpleNamespace(system_one=lambda **kwargs: (calls.append(kwargs), response)[1])
        fake_sdk = ModuleType("typesafe_sdk")
        fake_sdk.Noul = lambda **kwargs: ("noul", kwargs)
        fake_sdk.Choice = lambda **kwargs: ("choice", kwargs)
        engine = TypeSafeJevEngine(
            client,
            security_instructions="Does this request violate the configured policy?",
            price=Price(Decimal("0.05"), Decimal("0")),
        )
        cheap, costly = FakeProvider(), FakeProvider()
        events = []
        router = Router(
            jev=engine,
            economy=ModelTarget("cheap", "economy-model", Price(Decimal("1"), Decimal("2"))),
            frontier=ModelTarget("costly", "frontier-model", Price(Decimal("10"), Decimal("20"))),
            providers={"cheap": cheap, "costly": costly},
            on_event=events.append,
        )
        with patch.dict(sys.modules, {"typesafe_sdk": fake_sdk}):
            result = router.create(MESSAGES)
        self.assertEqual(len(calls), 1)
        self.assertEqual(set(calls[0]["questions"]), {"security", "complexity"})
        self.assertEqual(calls[0]["state"]["messages"], MESSAGES)
        self.assertEqual(result.decision.route, "economy")
        self.assertEqual(result.decision.jev_usage, Usage(300, 10))
        self.assertEqual(result.cost.router_overhead_usd, Decimal("0.000015"))
        self.assertEqual(events[0].router_cost_usd, Decimal("0.000015"))


if __name__ == "__main__":
    unittest.main()
