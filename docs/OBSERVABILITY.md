# Observability (Phase 6)

## Structured Logs
FastAPI emits JSON logs for key actions:
- `search` / `search_error`
- `ingest_single` / `ingest_single_error`
- `workflow` / `workflow_error`

Each log includes:
- `event`
- `request_id`
- `timestamp`
- relevant fields (query, top_k, duration_ms, etc.)

## Health Metrics Endpoint
`GET /api/health/metrics`

Returns:
```json
{
  "status": "ok",
  "timestamp": "...",
  "qdrant": {
    "available": true,
    "collections": ["molecules", "proteins"],
    "stats": { "molecules": { "points_count": 1234 } }
  },
  "models": {
    "available": true,
    "device": "cuda",
    "obm_loaded": true
  }
}
```

## CI-Style Test Runner
`python scripts/run_tests.py`

Runs:
- `test_search_api.py`
- `test_agent_api.py`
- `test_search_filters.py`
- `test_phase4_ui.py`
- `test_ingestion_api.py`

