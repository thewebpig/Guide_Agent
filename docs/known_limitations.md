# Known limitations

- This is a single-turn demo; it deliberately has no conversation memory or user authentication.
- The scene is synthetic and small. Replace `scenes/demo/scene.json` and its documents for a real venue.
- Tool calls require a model that supports OpenAI-style function calling. Providers may implement only one configured API format.
- MCP runs locally over stdio. A production deployment needs process supervision, authentication, telemetry, and rate limits.
