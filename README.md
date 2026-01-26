# 🧬 BioFlow Orchestrator

> **Multimodal Biological Design & Discovery Intelligence Engine**  
> A low-code workflow platform for unified biological discovery pipelines

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Next.js](https://img.shields.io/badge/Next.js-16-black)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green)
![Langflow](https://img.shields.io/badge/Langflow-1.7-orange)
![Team](https://img.shields.io/badge/Team-Lacoste-purple)

---

## 🎯 Problem Statement

Biological R&D knowledge is fragmented across disconnected silos:
- **Textual literature** (papers, lab notes)
- **3D structural data** (PDB files)
- **Chemical sequences** (SMILES)

Researchers must manually navigate incompatible formats, creating bottlenecks and "blind spots" where critical connections are missed.

## 💡 Our Solution

**BioFlow Orchestrator** is a visual workflow engine that unifies biological discovery pipelines. Rather than a single "black box" model, we function as an **intelligent orchestrator** — allowing researchers to chain state-of-the-art open-source biological models into coherent discovery workflows.

### Key Features

| Feature | Description |
|---------|-------------|
| 🔗 **Visual Pipeline Builder** | Drag-and-drop node editor for constructing discovery workflows |
| 🧠 **DeepPurpose Integration** | Drug-Target Interaction prediction with Morgan + CNN encoding |
| 🔍 **Qdrant Vector Search** | High-dimensional similarity search across 23,531+ compounds |
| 🎨 **3D Embedding Explorer** | Real PCA projections of drug-target chemical space |
| ✅ **Validator Agents** | Automated toxicity and novelty checking |

---

## 🏗️ Architecture

```
                         ┌──────────────────────────────────────────┐
                         │         BioFlow Orchestrator             │
                         │      Visual Pipeline Builder (UI)        │
                         └─────────────────┬────────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         │                                 │                                 │
         ▼                                 ▼                                 ▼
┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
│   Data Input    │             │   DeepPurpose   │             │   OpenBioMed    │
│  SMILES/Protein │────────────▶│   DTI Model     │────────────▶│   Multimodal    │
│   Sequences     │             │  Morgan + CNN   │             │   Embeddings    │
└─────────────────┘             └────────┬────────┘             └────────┬────────┘
                                         │                               │
                                         └───────────────┬───────────────┘
                                                         │
                                                         ▼
                                              ┌─────────────────┐
                                              │     Qdrant      │
                                              │   Vector DB     │
                                              │  HNSW Indexing  │
                                              │  23,531 vectors │
                                              └────────┬────────┘
                                                       │
                         ┌─────────────────────────────┼─────────────────────────────┐
                         │                             │                             │
                         ▼                             ▼                             ▼
              ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
              │ Similarity      │          │   Validator     │          │    Results      │
              │ Search Agent    │          │   Agent         │          │    Output       │
              │ Top-K Retrieval │          │ Toxicity/Novelty│          │   Candidates    │
              └─────────────────┘          └─────────────────┘          └─────────────────┘
```

---

## 📊 Model Performance

| Dataset | Concordance Index | Pearson | MSE |
|---------|-------------------|---------|-----|
| **KIBA** | 0.7003 | 0.5219 | 0.0008 |
| **BindingDB_Kd** | 0.8083 | 0.7679 | 0.6668 |
| **DAVIS** | 0.7914 | 0.5446 | 0.4684 |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker Desktop
- CUDA 11.8 (optional, for GPU acceleration)

### 1. Clone & Setup
```bash
git clone https://github.com/hamzasammoud11-dotcom/lacoste001.git
cd lacoste001

# Python environment
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install DeepPurpose qdrant-client fastapi uvicorn scikit-learn
```

### 2. Start Qdrant Vector Database
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

### 3. Ingest Data (One-time)
```bash
python ingest_qdrant.py
# Loads KIBA dataset → DeepPurpose embeddings → Qdrant
# ~23,531 drug-target pairs indexed
```

### 4. Start Backend API
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

### 6. Start Langflow (Visual Workflow Builder)
```bash
pip install langflow
langflow run --host 0.0.0.0 --port 7860
# Access via http://localhost:3000/workflow (embedded)
# Or directly at http://localhost:7860
```

---

## 🎨 Visual Workflow Builder (Langflow Integration)

BioFlow integrates **Langflow** as the visual workflow engine, providing a full-screen drag-and-drop pipeline builder accessible from `/workflow`.

### Building a DTI Pipeline in Langflow

1. **Import the Template Flow**:
   - Open Langflow (`/workflow` or `localhost:7860`)
   - Click "New Project" → "Import"
   - Load `langflow/bioflow_dti_pipeline.json`

2. **Configure the Pipeline**:
   - **Drug Input**: Enter SMILES string (e.g., `CC(=O)Nc1ccc(O)cc1`)
   - **Target Input**: Enter protein sequence
   - **API Nodes**: Point to `http://localhost:8001/api/*`

3. **Run the Flow**:
   - Click "Run" to execute DeepPurpose encoding → Qdrant search → Results

---

## 📁 Project Structure

```
├── config.py              # Shared configuration
├── ingest_qdrant.py       # ETL: TDC → DeepPurpose → Qdrant
├── deeppurpose002.py      # Model training script
├── server/
│   └── api.py             # FastAPI backend
├── runs/
│   └── 20260125_104915_KIBA/
│       ├── model.pt       # Trained model weights
│       └── config.pkl     # Model configuration
├── ui/
│   ├── app/
│   │   ├── workflow/      # 🆕 Visual Pipeline Builder
│   │   ├── explorer/      # 3D Embedding Visualization
│   │   ├── discovery/     # Drug Discovery Interface
│   │   └── data/          # Data Browser
│   └── components/
└── data/
    └── kiba.tab           # Cached TDC dataset
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health + model metrics |
| `/api/points` | GET | Get 3D PCA points for visualization |
| `/api/search` | POST | Similarity search by SMILES/sequence |

### Example: Search Similar Compounds
```bash
curl -X POST "http://localhost:8001/api/search" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Nc1ccc(O)cc1", "top_k": 10}'
```

---

## 🛠️ Qdrant Integration Strategy

### 1. Multimodal Bridge
Using OpenBioMed for joint embeddings across proteins, molecules, and text — enabling **cross-modal retrieval**.

### 2. Dynamic Workflow Memory
Pipeline nodes store intermediate results in Qdrant collections, enabling agent-to-agent communication.

### 3. High-Dimensional Scalability
HNSW indexing handles bio-embeddings at scale, keeping similarity searches interactive and real-time.

---

## 👥 Team Lacoste

| Name | Role |
|------|------|
| **Hamza Sammoud** | ML Pipeline & Backend |
| **Rami Troudi** | Frontend UI |

---

## 📚 Resources

- [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose) — DTI Prediction Toolkit
- [OpenBioMed](https://github.com/PharMolix/OpenBioMed) — Multimodal AI Framework
- [Qdrant](https://qdrant.tech/) — Vector Database
- [TDC](https://tdcommons.ai/) — Therapeutics Data Commons

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
