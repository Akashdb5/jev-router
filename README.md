# Jev Router

Jev-powered security screening and cost-aware routing for OpenAI, Anthropic, and OpenRouter LLMs.

Python-first SDK for prescreening and routing text LLM requests. The Python implementation lives in [`python/`](python/); [`spec/`](spec/) holds language-neutral routing fixtures for a future TypeScript SDK.

The SDK can batch Noul and Choice through the official TypeSafe Python client or OpenRouter's Decisions API in one Jev call. It sends high-confidence simple requests to the configured economy target and all other allowed requests to the frontier target. For evidence-backed answers, an optional verified cascade checks the draft and escalates once when needed. It records route latency and estimated cost without retaining prompt text in metrics events.

This first release handles synchronous, text-only requests. Streaming, tool calls, multimodal content, and the HTTP gateway are later milestones.

See the [Python README](python/README.md) for installation, usage, and current limits.

## License

Released under the [MIT License](LICENSE).
