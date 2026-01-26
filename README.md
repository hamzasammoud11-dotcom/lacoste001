# 🧬 Multimodal Biological Design & Discovery Intelligence

> Drug-Target Interaction prediction platform powered by DeepPurpose ML + Qdrant vector search + Next.js visualization

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Next.js](https://img.shields.io/badge/Next.js-16-black)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Overview

This platform enables **drug discovery** through:
- **Deep Learning** — Morgan fingerprints + CNN protein encoding (DeepPurpose)
- **Vector Search** — Similarity search via Qdrant embeddings
- **3D Visualization** — Real PCA projections of drug-target space
- **Interactive UI** — Next.js dashboard with Recharts

## 📊 Model Performance

| Dataset | Concordance Index | Pearson | MSE |
|---------|-------------------|---------|-----|
| **KIBA** | 0.7003 | 0.5219 | 0.0008 |
| **BindingDB_Kd** | 0.8083 | 0.7679 | 0.6668 |
| **DAVIS** | 0.7914 | 0.5446 | 0.4684 |

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   TDC Dataset   │────▶│   DeepPurpose   │────▶│     Qdrant      │
│   (KIBA/DAVIS)  │     │  Morgan + CNN   │     │  256D Vectors   │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              │
│   Next.js UI    │◀────│   FastAPI       │◀─────────────┘
│   localhost:3000│     │   localhost:8001│   Similarity Search
└─────────────────┘     └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10
- Node.js 18+
- Docker Desktop (for Qdrant)
- CUDA 11.8 (optional, for GPU)

### 1. Setup Python Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install DeepPurpose qdrant-client fastapi uvicorn scikit-learn
```

### 2. Start Qdrant (Vector Database)
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

### 3. Ingest Data (One-time)
```bash
python ingest_qdrant.py
# Loads KIBA dataset, generates embeddings, uploads to Qdrant
# ~23,531 drug-target pairs indexed
```

### 4. Start API Server
```bash
python -m uvicorn server.api:app --host 0.0.0.0 --port 8001
```

### 5. Start Frontend
```bash
cd ui
pnpm install
pnpm dev
# Open http://localhost:3000
```

## 📁 Project Structure

```
├── config.py              # Shared configuration (model paths, Qdrant settings)
├── ingest_qdrant.py       # ETL: TDC → DeepPurpose → Qdrant
├── deeppurpose002.py      # Model training script
├── server/
│   └── api.py             # FastAPI backend (/health, /api/points, /api/search)
├── runs/
│   └── 20260125_104915_KIBA/  # Best model checkpoint
│       ├── model.pt
│       └── config.pkl
├── ui/                    # Next.js 16 + Shadcn UI
│   ├── app/
│   │   ├── explorer/      # 3D scatter plot visualization
│   │   ├── discovery/     # Drug discovery interface
│   │   └── data/          # Data browser
│   └── lib/
│       └── explorer-service.ts  # API client
└── data/
    └── kiba.tab           # Cached TDC dataset
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health + metrics |
| `/api/points` | GET | Get 3D PCA points for visualization |
| `/api/search` | POST | Similarity search by SMILES/sequence |

### Example: Get Points
```bash
curl "http://localhost:8001/api/points?limit=100&view=combined"
```

### Example: Search Similar
```bash
curl -X POST "http://localhost:8001/api/search" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "top_k": 10}'
```

## 🧪 Training New Models

```bash
# Edit deeppurpose002.py to change dataset/encoding
python deeppurpose002.py

# Re-ingest with new model
python ingest_qdrant.py
```

## 👥 Contributors

- **Hamza Sammoud** — ML Pipeline & Backend
- **Rami Troudi** — Frontend UI

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

*Built for Hackathon 2026* 🏆
