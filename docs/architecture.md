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

Set `OPENAI_BASE_URL` to any OpenAI-compatible API root and select `OPENAI_API_FORMAT=responses` or `chat_completions`.
