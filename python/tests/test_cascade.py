import sys
import unittest
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_router import (  # noqa: E402
    Complexity,
    ComplexityAssessment,
    Completion,
    ModelTarget,
    Price,
    Router,
    SecurityAssessment,
    SupportLabel,
    TypeSafeJevVerifier,
    Usage,
    Verification,
    VerificationUnavailable,
    VerifiedCascade,
)


class StaticNoul:
    def evaluate(self, context):
        return SecurityAssessment(False, 0.99)


class StaticChoice:
    def __init__(self, label=Complexity.SIMPLE_LOOKUP):
        self.label = label

    def classify(self, context):
        return ComplexityAssessment(self.label, 0.99)


class FakeProvider:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return Completion(self.answer, Usage(100, 20), object())


class FakeVerifier:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def verify(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def verdict(label, confidence):
    return Verification(label, confidence, {label.value: 1.0}, Usage(50, 5), Decimal("0.00001"))


def make_cascade(verifier, label=Complexity.SIMPLE_LOOKUP):
    cheap, frontier = FakeProvider("cheap answer"), FakeProvider("frontier answer")
    events = []
    router = Router(
        noul=StaticNoul(), choice=StaticChoice(label),
        economy=ModelTarget("cheap", "cheap-model", Price(Decimal("1"), Decimal("2"))),
        frontier=ModelTarget("frontier", "frontier-model", Price(Decimal("10"), Decimal("20"))),
        providers={"cheap": cheap, "frontier": frontier},
        router_overhead_usd=Decimal("0.0001"),
    )
    return VerifiedCascade(router, verifier, on_event=events.append), cheap, frontier, events


class CascadeTests(unittest.TestCase):
    def test_supported_draft_is_sent_without_frontier(self):
        verifier = FakeVerifier(verdict(SupportLabel.SUPPORTED, 0.91))
        cascade, cheap, frontier, events = make_cascade(verifier)
        result = cascade.answer("How many seats?", ["The plan has five seats."])
        self.assertEqual(result.route, "send")
        self.assertEqual(result.answer, "cheap answer")
        self.assertEqual(len(cheap.calls), 1)
        self.assertFalse(frontier.calls)
        self.assertEqual(len(verifier.calls), 1)
        self.assertIn("The plan has five seats.", cheap.calls[0]["context"].screening_text)
        self.assertEqual(result.total_cost_usd, Decimal("0.00025"))
        self.assertEqual(result.estimated_frontier_baseline_usd, Decimal("0.0014"))
        self.assertEqual(events[0].verdicts, ("supported",))
        self.assertFalse(hasattr(events[0], "answer"))

    def test_unsupported_draft_escalates_and_verifies_frontier(self):
        verifier = FakeVerifier(
            verdict(SupportLabel.UNSUPPORTED, 0.95),
            verdict(SupportLabel.SUPPORTED, 0.92),
        )
        cascade, cheap, frontier, _ = make_cascade(verifier)
        result = cascade.answer("How many seats?", ["The plan has five seats."])
        self.assertEqual(result.route, "send")
        self.assertEqual(result.answer, "frontier answer")
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(len(cheap.calls), 1)
        self.assertEqual(len(frontier.calls), 1)
        self.assertEqual(len(verifier.calls), 2)
        self.assertLess(result.estimated_savings_usd, 0)

    def test_confident_decline_hands_off_without_escalation(self):
        cascade, _, frontier, _ = make_cascade(FakeVerifier(verdict(SupportLabel.DECLINED, 0.93)))
        result = cascade.answer("Do you support Jira?", ["The plan has five seats."])
        self.assertEqual(result.route, "handoff")
        self.assertFalse(frontier.calls)

    def test_low_confidence_support_escalates(self):
        cascade, _, frontier, _ = make_cascade(FakeVerifier(
            verdict(SupportLabel.SUPPORTED, 0.79),
            verdict(SupportLabel.SUPPORTED, 0.91),
        ))
        result = cascade.answer("How many seats?", ["The plan has five seats."])
        self.assertEqual(result.route, "send")
        self.assertEqual(len(frontier.calls), 1)

    def test_frontier_first_verification_can_handoff(self):
        cascade, cheap, frontier, _ = make_cascade(
            FakeVerifier(verdict(SupportLabel.UNSUPPORTED, 0.99)),
            label=Complexity.COMPLEX_MULTI_STEP,
        )
        result = cascade.answer("How many seats?", ["The plan has five seats."])
        self.assertEqual(result.route, "handoff")
        self.assertFalse(cheap.calls)
        self.assertEqual(len(frontier.calls), 1)

    def test_verifier_failure_cannot_send_answer(self):
        cascade, _, _, _ = make_cascade(FakeVerifier(RuntimeError("offline")))
        with self.assertRaises(VerificationUnavailable):
            cascade.answer("How many seats?", ["The plan has five seats."])

    def test_input_validation(self):
        cascade, _, _, _ = make_cascade(FakeVerifier(verdict(SupportLabel.SUPPORTED, 0.99)))
        with self.assertRaises(ValueError):
            cascade.answer("", ["one"])
        with self.assertRaises(ValueError):
            cascade.answer("question", [])
        with self.assertRaises(ValueError):
            cascade.answer("question", "a passage")

    def test_typesafe_verifier_passes_question_evidence_answer(self):
        calls = []
        response = SimpleNamespace(
            choices={"support": SimpleNamespace(
                choice="supported", confidence=0.91,
                probabilities={"supported": 0.94, "unsupported": 0.06, "declined": 0},
            )},
            usage=SimpleNamespace(input_tokens=200, output_tokens=10),
        )
        client = SimpleNamespace(system_one=lambda **kwargs: (calls.append(kwargs), response)[1])
        fake_sdk = ModuleType("typesafe_sdk")
        fake_sdk.Choice = lambda **kwargs: ("choice", kwargs)
        verifier = TypeSafeJevVerifier(client, price=Price(Decimal("0.05"), Decimal("0")))
        with patch.dict(sys.modules, {"typesafe_sdk": fake_sdk}):
            actual = verifier.verify(question="How many?", evidence=["Five."], answer="Five.")
        self.assertEqual(calls[0]["state"], {
            "question": "How many?", "evidence": ["Five."], "assistant_answer": "Five.",
        })
        self.assertEqual(actual.label, SupportLabel.SUPPORTED)
        self.assertEqual(actual.cost_usd, Decimal("0.00001"))


if __name__ == "__main__":
    unittest.main()
