"""One-call Noul + Choice integration for the official TypeSafe Python SDK."""

from __future__ import annotations

from decimal import Decimal

from .core import (
    Complexity,
    ComplexityAssessment,
    JevAssessment,
    Price,
    PromptContext,
    SecurityAssessment,
    Usage,
)


COMPLEXITY_INSTRUCTIONS = "What level of reasoning is needed to answer the user's request accurately?"
COMPLEXITY_CRITERIA = {
    Complexity.SIMPLE_LOOKUP.value: "Direct lookup, extraction, or short transformation with little reasoning.",
    Complexity.MODERATE_REASONING.value: "Several connected steps or some synthesis, but no extensive planning.",
    Complexity.COMPLEX_MULTI_STEP.value: "Multi-step reasoning, difficult synthesis, planning, or broad context integration.",
}


class TypeSafeJevEngine:
    """Evaluate both decisions in one ``TypeSafeClient.system_one`` call.

    The caller owns the client lifecycle and configures its API key and timeout.
    """

    def __init__(self, client: object, *, security_instructions: str, price: Price) -> None:
        if not security_instructions.strip():
            raise ValueError("security_instructions must describe the organization's policy")
        self.client = client
        self.security_instructions = security_instructions
        self.price = price

    def evaluate(self, context: PromptContext) -> JevAssessment:
        try:
            from typesafe_sdk import Choice, Noul
        except ImportError as exc:
            raise ImportError("install jev-router[jev] to use TypeSafeJevEngine") from exc

        response = self.client.system_one(
            state={"messages": list(context.messages)},
            questions={
                "security": Noul(instructions=self.security_instructions),
                "complexity": Choice(
                    instructions=COMPLEXITY_INSTRUCTIONS,
                    criteria=COMPLEXITY_CRITERIA,
                ),
            },
        )
        probability = float(response.nouls["security"].noul)
        violation = probability >= 0.5
        security = SecurityAssessment(violation, probability if violation else 1 - probability)
        answer = response.choices["complexity"]
        complexity = ComplexityAssessment(Complexity(answer.choice), float(answer.confidence))
        reported_usage = getattr(response, "usage", None)
        if reported_usage is None:
            usage = None
            cost_usd = Decimal("0")
        else:
            usage = Usage(reported_usage.input_tokens, reported_usage.output_tokens)
            cost_usd = (
                Decimal(usage.input_tokens) * self.price.input_per_million
                + Decimal(usage.output_tokens) * self.price.output_per_million
            ) / Decimal(1_000_000)
        return JevAssessment(security, complexity, usage, cost_usd)
