# Architecture

```
Browser / API client → FastAPI → LangChain create_agent + ChatOpenAI
                                      │ configurable Responses or Chat Completions
                                      ▼
                            OpenAI-compatible provider
                                      │ tool call
                                      ▼
                            MCP stdio client → MCP server
                                                  ├─ search_knowledge
                                                  ├─ lookup_poi
                                                  └─ plan_route
```

The FastAPI process never calls business tools directly. `LangChainGuideAgent` discovers them through `langchain-mcp-adapters`; the MCP client validates the tool allowlist and scene fingerprint. `ChatService` builds answers from recorded tool results and exposes only sources and tool envelopes.

The deployment-owned provider name, API root, and API format live in `config.yaml`.
The API key is injected with the startup parameter or `OPENAI_API_KEY`, or kept in the ignored
`config.local.yaml`; visitors never receive provider configuration or credentials.
