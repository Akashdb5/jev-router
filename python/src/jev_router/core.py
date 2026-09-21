"""Provider-independent routing, safety decisions, and cost accounting."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from time import perf_counter
from typing import Callable, Mapping, Protocol, Sequence


class Complexity(str, Enum):
    SIMPLE_LOOKUP = "SIMPLE_LOOKUP"
    MODERATE_REASONING = "MODERATE_REASONING"
    COMPLEX_MULTI_STEP = "COMPLEX_MULTI_STEP"


@dataclass(frozen=True)
class PromptContext:
    """Validated text conversation passed to Jev and a provider."""

    messages: tuple[dict[str, str], ...]

    @classmethod
    def from_messages(cls, messages: Sequence[Mapping[str, object]]) -> PromptContext:
        if not messages:
            raise ValueError("messages must contain at least one message")
        clean: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise TypeError("each message must be a mapping")
            role, content = message.get("role"), message.get("content")
            if role not in ("system", "user", "assistant") or not isinstance(content, str):
                raise ValueError("only text system, user, and assistant messages are supported")
            clean.append({"role": role, "content": content})
        if not any(message["role"] == "user" for message in clean):
            raise ValueError("messages must contain a user message")
        return cls(tuple(clean))

    @property
    def screening_text(self) -> str:
        """Full conversation text; integrations may use roles separately instead."""
        return "\n".join(f"{message['role']}: {message['content']}" for message in self.messages)


def _confidence(value: float) -> None:
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class SecurityAssessment:
    violation: bool
    confidence: float
    reason: str | None = None

    def __post_init__(self) -> None:
        _confidence(self.confidence)


@dataclass(frozen=True)
class ComplexityAssessment:
    label: Complexity
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, Complexity):
            raise TypeError("label must be a Complexity")
        _confidence(self.confidence)


class NoulGate(Protocol):
    def evaluate(self, context: PromptContext) -> SecurityAssessment: ...


class ChoiceClassifier(Protocol):
    def classify(self, context: PromptContext) -> ComplexityAssessment: ...


@dataclass(frozen=True)
class JevAssessment:
    """Noul and Choice answers from one Jev evaluation."""

    security: SecurityAssessment
    complexity: ComplexityAssessment
    usage: Usage | None = None
    cost_usd: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if self.cost_usd < 0:
            raise ValueError("Jev cost cannot be negative")


class JevDecisionEngine(Protocol):
    def evaluate(self, context: PromptContext) -> JevAssessment: ...


@dataclass(frozen=True)
class Price:
    """USD per one million input and output tokens, excluding discounts."""

    input_per_million: Decimal
    output_per_million: Decimal

    def __post_init__(self) -> None:
        if self.input_per_million < 0 or self.output_per_million < 0:
            raise ValueError("token prices cannot be negative")


@dataclass(frozen=True)
class ModelTarget:
    provider: str
    model: str
    price: Price


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token usage cannot be negative")


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage | None
    raw: object


class Provider(Protocol):
    def complete(
        self,
        *,
        model: str,
        context: PromptContext,
        max_tokens: int,
        temperature: float | None,
    ) -> Completion: ...


@dataclass(frozen=True)
class RouterPolicy:
    block_threshold: float = 0.95
    security_clear_threshold: float = 0.95
    economy_threshold: float = 0.88
    low_confidence_threshold: float = 0.70

    def __post_init__(self) -> None:
        for value in (self.block_threshold, self.security_clear_threshold, self.economy_threshold, self.low_confidence_threshold):
            _confidence(value)
        if self.low_confidence_threshold > self.economy_threshold:
            raise ValueError("low confidence threshold cannot exceed economy threshold")


@dataclass(frozen=True)
class RouteDecision:
    route: str  # economy, frontier, or blocked
    target: ModelTarget | None
    reason: str
    security: SecurityAssessment
    complexity: ComplexityAssessment | None
    gate_latency_ms: float
    jev_usage: Usage | None = None
    jev_cost_usd: Decimal = Decimal("0")


@dataclass(frozen=True)
class CostRecord:
    """Baseline uses selected-model token counts as a proxy for frontier usage."""

    estimated_frontier_baseline_usd: Decimal
    actual_provider_usd: Decimal
    router_overhead_usd: Decimal
    estimated_savings_usd: Decimal


@dataclass(frozen=True)
class RouterEvent:
    """Prompt-free event for metrics sinks."""

    route: str
    reason: str
    model: str | None
    provider: str | None
    gate_latency_ms: float
    total_latency_ms: float
    usage: Usage | None
    cost: CostRecord | None
    jev_usage: Usage | None = None
    router_cost_usd: Decimal = Decimal("0")


@dataclass(frozen=True)
class RoutedCompletion:
    completion: Completion
    decision: RouteDecision
    cost: CostRecord | None
    total_latency_ms: float

    @property
    def text(self) -> str:
        return self.completion.text


class PromptBlocked(Exception):
    def __init__(self, decision: RouteDecision):
        super().__init__("Noul blocked this prompt")
        self.decision = decision


class GateUnavailable(RuntimeError):
    """Noul did not return a valid security assessment; provider was not called."""


class Router:
    def __init__(
        self,
        *,
        economy: ModelTarget,
        frontier: ModelTarget,
        providers: Mapping[str, Provider],
        noul: NoulGate | None = None,
        choice: ChoiceClassifier | None = None,
        jev: JevDecisionEngine | None = None,
        policy: RouterPolicy | None = None,
        router_overhead_usd: Decimal = Decimal("0"),
        on_event: Callable[[RouterEvent], None] | None = None,
    ) -> None:
        if economy.provider not in providers or frontier.provider not in providers:
            raise ValueError("both model targets need a registered provider")
        if jev is None and (noul is None or choice is None):
            raise ValueError("supply one Jev engine or both Noul and Choice implementations")
        if router_overhead_usd < 0:
            raise ValueError("router overhead cannot be negative")
        self.noul = noul
        self.choice = choice
        self.jev = jev
        self.economy = economy
        self.frontier = frontier
        self.providers = dict(providers)
        self.policy = policy or RouterPolicy()
        self.router_overhead_usd = router_overhead_usd
        self.on_event = on_event

    def decide(self, context: PromptContext) -> RouteDecision:
        start = perf_counter()
        jev_usage: Usage | None = None
        jev_cost_usd = Decimal("0")
        if self.jev is not None:
            try:
                combined = self.jev.evaluate(context)
                if not isinstance(combined, JevAssessment):
                    raise TypeError("Jev returned an invalid assessment")
                security = combined.security
                complexity = combined.complexity
                jev_usage = combined.usage
                jev_cost_usd = combined.cost_usd
            except Exception as exc:
                raise GateUnavailable("Jev could not screen the request") from exc
        else:
            assert self.noul is not None and self.choice is not None
            try:
                security = self.noul.evaluate(context)
                if not isinstance(security, SecurityAssessment):
                    raise TypeError("Noul returned an invalid assessment")
            except Exception as exc:
                raise GateUnavailable("Noul could not screen the request") from exc
            complexity = None

        if security.violation and security.confidence >= self.policy.block_threshold:
            return RouteDecision(
                "blocked", None, "high_confidence_violation", security, None,
                (perf_counter() - start) * 1000, jev_usage, jev_cost_usd,
            )

        if complexity is None:
            try:
                assert self.choice is not None
                complexity = self.choice.classify(context)
                if not isinstance(complexity, ComplexityAssessment):
                    raise TypeError("Choice returned an invalid assessment")
            except Exception:
                return RouteDecision(
                    "frontier", self.frontier, "choice_unavailable", security, None,
                    (perf_counter() - start) * 1000,
                )

        if security.violation:
            route, target, reason = "frontier", self.frontier, "uncertain_violation"
        elif security.confidence < self.policy.security_clear_threshold:
            route, target, reason = "frontier", self.frontier, "uncertain_security_clearance"
        elif complexity.confidence < self.policy.low_confidence_threshold:
            route, target, reason = "frontier", self.frontier, "low_choice_confidence"
        elif complexity.label == Complexity.SIMPLE_LOOKUP and complexity.confidence >= self.policy.economy_threshold:
            route, target, reason = "economy", self.economy, "high_confidence_simple"
        else:
            route, target, reason = "frontier", self.frontier, "not_high_confidence_simple"
        return RouteDecision(route, target, reason, security, complexity, (perf_counter() - start) * 1000, jev_usage, jev_cost_usd)

    def create(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> RoutedCompletion:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        context = PromptContext.from_messages(messages)
        start = perf_counter()
        decision = self.decide(context)
        if decision.route == "blocked":
            self._emit(RouterEvent(
                "blocked", decision.reason, None, None, decision.gate_latency_ms,
                (perf_counter() - start) * 1000, None, None,
                decision.jev_usage, self.router_overhead_usd + decision.jev_cost_usd,
            ))
            raise PromptBlocked(decision)
        assert decision.target is not None
        target = decision.target
        completion = self.providers[target.provider].complete(
            model=target.model,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        cost = self._cost(target, completion.usage, decision.jev_cost_usd) if completion.usage else None
        total_latency_ms = (perf_counter() - start) * 1000
        self._emit(RouterEvent(
            decision.route, decision.reason, target.model, target.provider,
            decision.gate_latency_ms, total_latency_ms, completion.usage, cost,
            decision.jev_usage, self.router_overhead_usd + decision.jev_cost_usd,
        ))
        return RoutedCompletion(completion, decision, cost, total_latency_ms)

    def _cost(self, target: ModelTarget, usage: Usage, jev_cost_usd: Decimal) -> CostRecord:
        million = Decimal(1_000_000)
        input_tokens, output_tokens = Decimal(usage.input_tokens), Decimal(usage.output_tokens)
        baseline = (
            input_tokens * self.frontier.price.input_per_million
            + output_tokens * self.frontier.price.output_per_million
        ) / million
        actual = (
            input_tokens * target.price.input_per_million
            + output_tokens * target.price.output_per_million
        ) / million
        overhead = self.router_overhead_usd + jev_cost_usd
        return CostRecord(baseline, actual, overhead, baseline - actual - overhead)

    def _emit(self, event: RouterEvent) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                # A metrics sink must not change the result of an LLM request.
                pass
