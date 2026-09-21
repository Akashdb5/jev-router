"""Adapters for official OpenAI and Anthropic Python client objects."""

from __future__ import annotations

from .core import Completion, PromptContext, Usage


def _usage(source: object, input_name: str, output_name: str) -> Usage | None:
    if source is None:
        return None
    input_tokens = getattr(source, input_name, None)
    output_tokens = getattr(source, output_name, None)
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    return Usage(input_tokens, output_tokens)


class OpenAIChatProvider:
    """Pass an initialized ``openai.OpenAI`` client to this adapter."""

    def __init__(self, client: object) -> None:
        self.client = client

    def complete(
        self,
        *,
        model: str,
        context: PromptContext,
        max_tokens: int,
        temperature: float | None,
    ) -> Completion:
        request: dict[str, object] = {
            "model": model,
            "messages": list(context.messages),
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            request["temperature"] = temperature
        response = self.client.chat.completions.create(**request)
        text = response.choices[0].message.content or ""
        return Completion(text, _usage(getattr(response, "usage", None), "prompt_tokens", "completion_tokens"), response)


class AnthropicMessagesProvider:
    """Pass an initialized ``anthropic.Anthropic`` client to this adapter."""

    def __init__(self, client: object) -> None:
        self.client = client

    def complete(
        self,
        *,
        model: str,
        context: PromptContext,
        max_tokens: int,
        temperature: float | None,
    ) -> Completion:
        system: list[str] = []
        messages: list[dict[str, str]] = []
        for message in context.messages:
            if message["role"] == "system":
                if messages:
                    raise ValueError("Anthropic requires system messages before user and assistant messages")
                system.append(message["content"])
            else:
                messages.append(message)
        request: dict[str, object] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            request["system"] = "\n\n".join(system)
        if temperature is not None:
            request["temperature"] = temperature
        response = self.client.messages.create(**request)
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return Completion(text, _usage(getattr(response, "usage", None), "input_tokens", "output_tokens"), response)
