# BioFlow

> **Multimodal Biological Design & Discovery Intelligence Engine**  
> A full-stack AI platform for unified biological discovery pipelines

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Next.js](https://img.shields.io/badge/Next.js-16-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Qdrant Cloud](https://img.shields.io/badge/Qdrant-Cloud-red)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)
![Team](https://img.shields.io/badge/Team-Lacoste-purple)

🚀 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/vignt97867896/bioflow)**

---

## 🧬 What is BioFlow?

**BioFlow** is a comprehensive AI-powered platform for drug discovery and biological research. It combines:

- **Drug-Target Interaction (DTI) Prediction** - Deep learning models for binding affinity prediction
- **Vector Similarity Search** - Qdrant Cloud-powered semantic search across 23,000+ compounds
- **3D Molecular Visualization** - Interactive 3D viewers for molecules and proteins
- **Visual Workflow Builder** - Langflow-powered drag-and-drop pipeline construction
- **Multi-Agent System** - Generator, Ranker, and Validator agents for compound discovery

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🔬 DTI Prediction** | DeepPurpose Morgan+CNN encoder for drug-target binding affinity |
| **🔍 Semantic Search** | Qdrant Cloud vector search with MMR diversity sampling |
| **🧪 Molecule Viewer** | 2D SMILES rendering + 3D molecular structure (3Dmol.js) |
| **🧬 Protein Viewer** | Interactive PDB structure visualization |
| **📊 3D Explorer** | Real PCA projections of chemical embedding space |
| **🤖 AI Agents** | Generator → Ranker → Validator pipeline for discovery |
| **🔧 Visual Workflows** | Langflow integration for custom pipelines |
| **☁️ Cloud-Native** | Qdrant Cloud + Hugging Face Spaces deployment |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BioFlow Platform                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│   │   Next.js 16     │    │   FastAPI        │    │   Qdrant Cloud   │  │
│   │   Frontend       │◄──►│   Backend        │◄──►│   Vector DB      │  │
│   │   (TypeScript)   │    │   (Python)       │    │   (23K+ vectors) │  │
│   └──────────────────┘    └──────────────────┘    └──────────────────┘  │
│           │                       │                                      │
│           ▼                       ▼                                      │
│   ┌──────────────────┐    ┌──────────────────┐                          │
│   │   Dashboard      │    │   AI Modules     │                          │
│   │   • Discovery    │    │   • DeepPurpose  │                          │
│   │   • Explorer     │    │   • OpenBioMed   │                          │
│   │   • Molecules    │    │   • Encoders     │                          │
│   │   • Proteins     │    │   • Agents       │                          │
│   │   • Workflow     │    │   • Search       │                          │
│   └──────────────────┘    └──────────────────┘                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Model Performance

| Dataset | Concordance Index | Pearson Correlation | MSE |
|---------|-------------------|---------------------|-----|
| **DAVIS** | 0.7914 | 0.5446 | 0.4684 |
| **KIBA** | 0.7003 | 0.5219 | 0.0008 |
| **BindingDB_Kd** | 0.8083 | 0.7679 | 0.6668 |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+ (with pnpm)
- Git

### 1. Clone & Setup

```bash
git clone https://github.com/hamzasammoud11-dotcom/lacoste001.git
cd lacoste001

# Create Python environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file (or copy from template):

```env
# Qdrant Cloud (Required)
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-api-key

# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
```

> **Note**: We use **Qdrant Cloud** instead of local Docker for production reliability.
> Get free credentials at [cloud.qdrant.io](https://cloud.qdrant.io)

### 3. Start Backend

```bash
python -m uvicorn bioflow.api.server:app --host 0.0.0.0 --port 8000
```

### 4. Start Frontend

```bash
cd ui
pnpm install
pnpm dev
```

### 5. Start Langflow (Visual Workflow Builder)
```bash
# You can use the provided script
python run_langflow.py

# Or manually:
pip install langflow
langflow run --host 0.0.0.0 --port 7860
# Access via http://localhost:3000/workflow (embedded)
# Or directly at http://localhost:7860
```

**Note:** To use a dedicated Python environment for Langflow without installing it in the project venv, set the `LANGFLOW_PYTHON` environment variable to the path of the Python executable that has Langflow installed:

```bash
LANGFLOW_PYTHON=C:\path\to\python.exe python run_langflow.py
```

Open [http://localhost:3000](http://localhost:3000)

---

## 📂 Project Structure

```
bioflow/
├── .env                    # Environment configuration (Qdrant, API URLs)
├── config.py               # Shared Python configuration
├── requirements.txt        # Python dependencies
│
├── bioflow/                # Core Python package
│   ├── api/                # FastAPI backend
│   │   ├── server.py       # Main API server
│   │   ├── dti_predictor.py
│   │   └── qdrant_service.py
│   ├── agents/             # Multi-agent system
│   │   ├── generator.py    # Compound generation
│   │   ├── ranker.py       # Affinity ranking
│   │   └── validator.py    # Safety validation
│   ├── core/               # Base classes & orchestration
│   ├── plugins/            # DeepPurpose, OBM encoders
│   ├── search/             # Enhanced search (MMR, filters)
│   └── ingestion/          # Data ingestion pipelines
│
├── ui/                     # Next.js 16 Frontend
│   ├── app/
│   │   ├── dashboard/
│   │   │   ├── discovery/      # Drug discovery interface
│   │   │   ├── explorer/       # 3D embedding visualization
│   │   │   ├── molecules-2d/   # SMILES 2D viewer
│   │   │   ├── molecules-3d/   # 3Dmol.js 3D viewer
│   │   │   ├── proteins-3d/    # Protein structure viewer
│   │   │   ├── workflow/       # Visual pipeline builder
│   │   │   └── data/           # Data browser
│   │   └── api/            # Next.js API routes (proxy)
│   └── components/         # Reusable UI components
│
├── tests/                  # Test suite
│   ├── test_agents.py
│   ├── test_search_api.py
│   └── stress_test_uc4.py
│
├── runs/                   # Model training outputs
├── data/                   # Cached datasets (KIBA, DAVIS)
└── docs/                   # Documentation
```

---

## 🔌 API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health + model status |
| `/api/points` | GET | 3D PCA embedding coordinates |
| `/api/search` | POST | Similarity search by SMILES |
| `/api/molecules` | GET | Browse indexed molecules |
| `/api/proteins` | GET | Browse indexed proteins |
| `/api/predict` | POST | DTI binding prediction |

### Agent Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agents/generate` | POST | Generate candidate molecules |
| `/api/agents/rank` | POST | Rank by binding affinity |
| `/api/agents/validate` | POST | Validate safety/novelty |
| `/api/agents/workflow` | POST | Run full agent pipeline |

### Example: Search Similar Compounds

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Nc1ccc(O)cc1", "top_k": 10}'
```

---

## ☁️ Deployment

### Hugging Face Spaces (Backend)
The FastAPI backend is deployed on HF Spaces with Docker.

### Vercel (Frontend)
The Next.js frontend can be deployed to Vercel with:
```bash
cd ui
vercel --prod
```

### Environment Variables for Production

**Backend (.env)**:
```env
QDRANT_URL=https://xxx.cloud.qdrant.io
QDRANT_API_KEY=your-key
```

**Frontend (Vercel Dashboard)**:
```env
NEXT_PUBLIC_API_URL=https://your-space.hf.space
```

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python tests/test_search_api.py

# Stress test
python tests/stress_test_uc4.py
```

---

## 📚 Resources

- [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose) - DTI Prediction Toolkit
- [OpenBioMed](https://github.com/PharMolix/OpenBioMed) - Multimodal Bio-AI
- [Qdrant](https://qdrant.tech/) - Vector Database
- [3Dmol.js](https://3dmol.csb.pitt.edu/) - Molecular Visualization
- [Langflow](https://langflow.org/) - Visual LLM Workflows

---

## 👥 Team Lacoste

Built with ❤️ for biological discovery.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
