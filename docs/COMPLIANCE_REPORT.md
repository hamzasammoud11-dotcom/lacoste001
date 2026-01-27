# BioFlow Compliance Report (Phase 1)

**Date:** 2026-01-27  
**Scope:** Open-source compliance + forbidden runtime dependencies

## Summary
- **Streamlit UI removed** from runtime and repository path.
- **OpenAI / Azure OpenAI / Anthropic UI references removed**.
- **Runtime stack** confirmed: FastAPI + Next.js + Qdrant only.

## Removed / Deprecated
- `bioflow/app.py` (legacy Streamlit app) **deleted**
- `bioflow/ui/*` (Streamlit UI package) **deleted**
- Streamlit dependency **removed** from runtime requirements
- UI settings no longer expose proprietary LLM providers

## Allowed / Kept
- **OBM (OpenBioMed)** for embeddings only
- **DeepPurpose** for DTI (open-source)
- **Qdrant** as primary vector database

## Dependencies (Runtime)
From `requirements.txt` (open-source only):
- `fastapi`, `uvicorn`
- `qdrant-client`
- `torch`, `transformers`, `rdkit`, `numpy`, `scikit-learn`
- `requests`, `pandas`, `dotenv`

## Remaining Risks / Follow-ups
- **Legacy references in docs** should avoid implying Streamlit runtime.
- Ensure **no proprietary endpoints** are configured in deployment.

## Evidence
- Streamlit files removed: `bioflow/app.py`, `bioflow/ui/*`
- UI settings updated: `ui/app/dashboard/settings/page.tsx`

