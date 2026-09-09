# Known limitations

- This is a single-turn demo; it deliberately has no conversation memory or user authentication.
- The scene is synthetic and small. Replace `scenes/demo/scene.json` and its documents for a real venue.
- The default RAG acceptance threshold (`0.50`) is calibrated only for the bundled demo scene and `BAAI/bge-small-zh-v1.5`. After Markdown semantic chunking, the bundled opening-hours question scored `0.715108`; the no-answer parking question scored `0.424282`, and the overly short query `停车` scored `0.496596`. Re-evaluate it with representative positive and no-answer questions whenever documents or embedding models change.
- Tool calls require a model that supports OpenAI-style function calling. Providers may implement only one configured API format.
- MCP runs locally over stdio. A production deployment needs process supervision, authentication, telemetry, and rate limits.
