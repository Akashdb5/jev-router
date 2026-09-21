"""Jev verification of answers against retrieved evidence."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Mapping, Protocol, Sequence

from .core import Price, Usage


class SupportLabel(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    DECLINED = "declined"


SUPPORT_INSTRUCTIONS = (
    "Compare the assistant answer with the evidence and the question. "
    "Which option accurately describes the answer?"
)
SUPPORT_CRITERIA = {
    "supported": "The answer addresses the question, and every factual claim, number, and policy in it is supported by the evidence.",
    "unsupported": "The answer makes a claim not supported by the evidence, contradicts it, or answers a different question.",
    "declined": "The answer says the evidence does not cover the question and makes no unsupported factual claims.",
}


@dataclass(frozen=True)
class Verification:
    label: SupportLabel
    confidence: float
    probabilities: Mapping[str, float]
    usage: Usage | None
    cost_usd: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.label, SupportLabel) or not 0 <= self.confidence <= 1:
            raise ValueError("invalid Jev verification answer")
        if self.cost_usd < 0:
            raise ValueError("verification cost cannot be negative")


class AnswerVerifier(Protocol):
    def verify(self, *, question: str, evidence: Sequence[str], answer: str) -> Verification: ...


class VerificationUnavailable(RuntimeError):
    """The answer could not be verified and should not be sent automatically."""


class TypeSafeJevVerifier:
    """Use an initialized TypeSafeClient for evidence-backed answer checks."""

    def __init__(self, client: object, *, price: Price) -> None:
        self.client = client
        self.price = price

    def verify(self, *, question: str, evidence: Sequence[str], answer: str) -> Verification:
        from typesafe_sdk import Choice

        response = self.client.system_one(
            state={"question": question, "evidence": list(evidence), "assistant_answer": answer},
            questions={
                "support": Choice(instructions=SUPPORT_INSTRUCTIONS, criteria=SUPPORT_CRITERIA),
            },
        )
        verdict = response.choices["support"]
        usage_object = getattr(response, "usage", None)
        usage = Usage(usage_object.input_tokens, usage_object.output_tokens) if usage_object is not None else None
        cost = (
            (
                Decimal(usage.input_tokens) * self.price.input_per_million
                + Decimal(usage.output_tokens) * self.price.output_per_million
            ) / Decimal(1_000_000)
            if usage is not None else Decimal("0")
        )
        return Verification(
            SupportLabel(verdict.choice), float(verdict.confidence),
            dict(verdict.probabilities), usage, cost,
        )
