# BioFlow Scope Lock (Phase 0)

**Status:** Confirmed by user  
**Date:** 2026-01-27

## Single Source of Truth Runtime
- **Backend:** FastAPI on port `8000`
- **Frontend:** Next.js on port `3000`
- **Vector DB:** Qdrant on port `6333`
- **Canonical pipeline:** Agents + Enhanced Search
- **Legacy:** Streamlit + legacy pipeline are deprecated and must not be on the runtime path

## Open‑Source Only Constraint
- **No proprietary dependencies** (OpenAI / Azure OpenAI / InstaDeep / closed models)
- **All referenced models must be open‑source**

## OBM Role
- OBM is the **multimodal embedding backbone only** (text / SMILES / protein)
- OBM is **not** the generator, validator, or orchestrator

## Qdrant Requirement
Qdrant must be implemented **to the fullest extent needed** for the project, including:
- **HNSW indexing** for scalable similarity search
- **Payload metadata + filtering** for evidence, modality, source, organism, dates
- **Collections** and **collection discovery** for multi‑source datasets
- **Efficient paging/scrolling** for UI listings and explorer views
- **Multi‑vector or named vectors** where required by modality
- **Top‑K filtered retrieval** and **context injection** into agents

## Audit Deliverable (Phase 0 Definition of Done)
The “Full Audit” report must include:
1. **Checklist vs requirements**
2. **Gaps & risks**
3. **Performance bottlenecks**
4. **Security/compliance review**
5. **Prioritized remediation plan**
6. **Appendix: API + data schemas**

