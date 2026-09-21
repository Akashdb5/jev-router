# Jev Router

Jev-powered security screening and cost-aware routing for OpenAI, Anthropic, and OpenRouter LLMs.

Python-first SDK for prescreening and routing text LLM requests. The Python implementation lives in [`python/`](python/); [`spec/`](spec/) holds language-neutral routing fixtures for a future TypeScript SDK.

The SDK can batch Noul and Choice through the official TypeSafe Python client or OpenRouter's Decisions API in one Jev call. It sends high-confidence simple requests to the configured economy target and all other allowed requests to the frontier target. For evidence-backed answers, an optional verified cascade checks the draft and escalates once when needed. It records route latency and estimated cost without retaining prompt text in metrics events.

## Observed example

In one live simple-lookup comparison, Jev routed `What is a ZIP code?` to GPT-4o Mini. The routed request cost `$0.000144978` including Jev, while a separate fixed GPT-4o call cost `$0.0010525`: an observed reduction of **86.2%**. The routed call took `5,272 ms` versus `1,881 ms` for the fixed call, so this example improved cost while increasing latency. This is one run, not a representative benchmark. See the [full method, token counts, calculation, and limitations](python/README.md#observed-cost-and-latency-example).

This first release handles synchronous, text-only requests. Streaming, tool calls, multimodal content, and the HTTP gateway are later milestones.

See the [Python README](python/README.md) for installation, usage, and current limits.

## License

Released under the [MIT License](LICENSE).
