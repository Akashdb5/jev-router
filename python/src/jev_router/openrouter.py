"""Optional Jev transport through OpenRouter's Decisions API."""

from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import Mapping, Sequence
from urllib.request import Request, urlopen

from .core import Complexity, ComplexityAssessment, JevAssessment, PromptContext, SecurityAssessment, Usage
from .jev import COMPLEXITY_CRITERIA, COMPLEXITY_INSTRUCTIONS
from .verify import SUPPORT_CRITERIA, SUPPORT_INSTRUCTIONS, SupportLabel, Verification


class OpenRouterDecisions:
    """Small synchronous client for ``POST /api/alpha/decisions``.

    Credentials come from ``OPENROUTER_API_KEY``. A caller can choose the
    model, endpoint base, and timeout at initialization.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://openrouter.ai",
        timeout: float = 10.0,
    ) -> None:
        if not model.strip() or timeout <= 0:
            raise ValueError("a model and positive timeout are required")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def evaluate(self, *, state: object, questions: Mapping[str, object]) -> dict:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter Decisions")
        payload = json.dumps({"model": self.model, "state": state, "questions": questions}).encode("utf-8")
        request = Request(
            self.base_url + "/api/alpha/decisions",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("OpenRouter Decisions response is too large")
        result = json.loads(raw, parse_float=Decimal)
        if not isinstance(result, dict):
            raise ValueError("OpenRouter Decisions returned an invalid response")
        return result


def _openrouter_usage(data: dict) -> tuple[Usage, Decimal]:
    reported = data["usage"]
    usage = Usage(reported["input_tokens"], reported["output_tokens"])
    # OpenRouter reports the billed amount; do not replace a missing amount with zero.
    cost = Decimal(str(reported["cost"]))
    if not cost.is_finite() or cost < 0:
        raise ValueError("OpenRouter returned an invalid cost")
    return usage, cost


class OpenRouterJevEngine:
    def __init__(self, decisions: OpenRouterDecisions, *, security_instructions: str) -> None:
        if not security_instructions.strip():
            raise ValueError("security_instructions must describe the organization's policy")
        self.decisions = decisions
        self.security_instructions = security_instructions

    def evaluate(self, context: PromptContext) -> JevAssessment:
        data = self.decisions.evaluate(
            state={"messages": list(context.messages)},
            questions={
                "security": {"type": "noul", "instructions": self.security_instructions},
                "complexity": {
                    "type": "choice", "instructions": COMPLEXITY_INSTRUCTIONS,
                    "criteria": COMPLEXITY_CRITERIA,
                },
            },
        )
        probability = float(data["answers"]["security"]["noul"])
        violation = probability >= 0.5
        security = SecurityAssessment(violation, probability if violation else 1 - probability)
        answer = data["answers"]["complexity"]
        complexity = ComplexityAssessment(Complexity(answer["choice"]), float(answer["confidence"]))
        usage, cost = _openrouter_usage(data)
        return JevAssessment(security, complexity, usage, cost)


class OpenRouterJevVerifier:
    def __init__(self, decisions: OpenRouterDecisions) -> None:
        self.decisions = decisions

    def verify(self, *, question: str, evidence: Sequence[str], answer: str) -> Verification:
        data = self.decisions.evaluate(
            state={"question": question, "evidence": list(evidence), "assistant_answer": answer},
            questions={
                "support": {
                    "type": "choice", "instructions": SUPPORT_INSTRUCTIONS,
                    "criteria": SUPPORT_CRITERIA,
                },
            },
        )
        verdict = data["answers"]["support"]
        usage, cost = _openrouter_usage(data)
        return Verification(
            SupportLabel(verdict["choice"]), float(verdict["confidence"]),
            dict(verdict["probabilities"]), usage, cost,
        )
