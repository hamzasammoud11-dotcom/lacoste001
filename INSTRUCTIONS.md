# Hackathon Setup: DeepPurpose + Qdrant + UI (v2)

## Architecture Overview
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Next.js UI │────▶│  FastAPI     │────▶│   Qdrant    │
│  (3000)     │     │  (8000)      │     │   (6333)    │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │ DeepPurpose │
                    │   Model     │
                    └─────────────┘
```

## Quick Start

### 1. Start Qdrant (Vector Database)
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Ingest Data (Phase 1)
```bash
.venv\Scripts\python ingest_qdrant.py
```
This will:
- Load the trained model from `runs/20260125_104915_KIBA`
- Generate embeddings for all drug-target pairs
- Compute **real PCA** projections (not fake first-3-dims!)
- Upload to Qdrant with pre-computed 3D coordinates

### 3. Start API Server (Phase 2)
```bash
.venv\Scripts\python server/api.py
```
API endpoints:
- `GET /api/points?limit=500&view=combined` - Get 3D visualization data
- `POST /api/search` - Find similar drugs/targets
- `GET /health` - Check system status

### 4. Start Frontend (Phase 3)
```bash
cd ui && pnpm dev
```
Open http://localhost:3000/explorer

## What's Fixed (v2)

| Issue | Before | After |
|-------|--------|-------|
| PCA | First 3 dims (meaningless) | Real sklearn PCA |
| Data Order | Shuffled (broken alignment) | `shuffle=False` in DataLoader |
| Dummy Data | `"M" * 10` (fragile) | Valid Aspirin SMILES |
| Config | Duplicated | Shared `config.py` |
| Error Handling | None | Validation + helpful messages |
| Model Loading | Per-request | Cached at startup |

## Best Model Results (Kept Runs)

| Dataset | CI | Pearson | Has model.pt |
|---------|-------|---------|--------------|
| BindingDB_Kd | **0.8083** | 0.7679 | No |
| DAVIS | 0.7914 | 0.5446 | No |
| KIBA | 0.7003 | 0.5219 | **Yes** |

*Note: KIBA run has the saved model, but BindingDB has best metrics.*

