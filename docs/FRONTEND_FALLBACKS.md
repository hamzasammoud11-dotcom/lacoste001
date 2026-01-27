# Frontend Fallback Behavior (Phase 5)

## Overview
The UI uses Next.js `/api/*` route handlers as a proxy layer to the FastAPI backend.  
If the backend is unavailable, these routes return **safe defaults** so the UI remains usable.

## Proxy Routes (Next.js)
All routes forward to `API_CONFIG.baseUrl` (`NEXT_PUBLIC_API_URL`, default `http://localhost:8000`).

### Search
- `POST /api/search`
- `POST /api/search/hybrid`
- Fallback: empty results with metadata stubs

### Agents
- `POST /api/agents/generate`
- `POST /api/agents/validate`
- `POST /api/agents/rank`
- `POST /api/agents/workflow`
- Fallback: empty payloads with `mock: true` flags (where applicable)

### Explorer
- `GET /api/explorer/embeddings`
- Fallback: `503` + empty points

### Ingestion
- `POST /api/ingest/pubmed`
- `POST /api/ingest/uniprot`
- `POST /api/ingest/chembl`
- `POST /api/ingest/all`
- `GET /api/ingest/jobs/{job_id}`
- Fallback: `503` + error message

## UI Empty‑State Handling
- Visualization page shows a **“No points to display”** message until a search runs.
- Workflow page shows **“Run workflow to see results”** until execution completes.

## Recommendation
For demos, keep the FastAPI backend running to ensure real data/embeddings are shown.

