from prometheus_client import Counter, Gauge

anomaly_events_total = Counter(
    "airsentinel_anomaly_events_total",
    "Total number of anomaly events pushed via WebSocket",
    ["location_id"],
)

websocket_connections_active = Gauge(
    "airsentinel_websocket_connections_active",
    "Number of currently active WebSocket connections",
)

llm_requests_total = Counter(
    "airsentinel_llm_requests_total",
    "Total LLM requests made by the agent",
    ["provider", "status"],
)
