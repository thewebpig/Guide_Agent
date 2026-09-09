# Three-minute demo

1. Start with `./scripts/start_demo.ps1 -ConfigureModel -ApiFormat chat_completions` and open the local URL.
2. Point out the fixed LangChain Agent and MCP stdio badges, plus the selected API format.
3. Ask “从主入口到人工智能实验室怎么走？”. Show the `plan_route` MCP trace and deterministic route answer.
4. Ask “体验中心常规开放时间是什么？”. Show `search_knowledge`, its evidence source, and similarity score.
5. Open `/health` to demonstrate safe configuration observability without exposing credentials.
