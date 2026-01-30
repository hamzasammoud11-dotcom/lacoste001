# BioFlow 🧬

> **AI-Powered Multimodal Drug Discovery & Biological Intelligence Platform**  
> Unifying molecules, proteins, text, and images for accelerated therapeutic research

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6?logo=typescript)
![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-DC382D)
![License](https://img.shields.io/badge/License-MIT-green)
![Team](https://img.shields.io/badge/Team-Lacoste-purple)

🚀 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/vignt97867896/bioflow)**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Model Performance](#-model-performance)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Dashboard Modules](#-dashboard-modules)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**BioFlow** is a comprehensive, open-source AI platform designed for modern drug discovery workflows. It bridges the gap between fragmented biological data sources by providing:

- **Unified Multimodal Search** - Query across molecules, proteins, scientific text, and biomedical images
- **Drug-Target Interaction (DTI) Prediction** - Deep learning models for binding affinity prediction
- **Intelligent Design Assistance** - AI-powered molecular variant suggestions with evidence chains
- **Interactive 3D Visualization** - Explore chemical embedding spaces and molecular structures
- **Evidence Traceability** - Link discoveries back to source experiments and publications

### Why BioFlow?

| Traditional Workflow | With BioFlow |
|---------------------|--------------|
| Siloed data across databases | Unified vector search across all modalities |
| Manual literature review | AI-powered evidence linking |
| Trial-and-error design | Data-driven variant suggestions |
| Disconnected tools | Integrated discovery pipeline |

---

## ✨ Key Features

### 🔬 Drug-Target Interaction Prediction
- **DeepPurpose Morgan+CNN** encoder for binding affinity prediction
- Support for DAVIS, KIBA, and BindingDB datasets
- Real-time IC50/Kd predictions with confidence scores

### 🔍 Multimodal Vector Search
- **Qdrant Cloud** powered semantic search across 23,000+ indexed compounds
- Maximum Marginal Relevance (MMR) for diversity in results
- Cross-modal queries: text → molecules, proteins → compounds

### 🧪 Molecular Visualization
- **2D SMILES Viewer** - Interactive chemical structure rendering
- **3D Molecular Viewer** - 3Dmol.js powered 3D structure visualization
- **Protein Structure Viewer** - Interactive PDB/AlphaFold structure viewing

### 📊 3D Embedding Explorer
- Real PCA projections of chemical embedding space
- Clustered visualization of compound relationships
- Interactive filtering by affinity, modality, and source

### 🤖 Multi-Agent Discovery System
- **Generator Agent** - Creates novel molecular variants
- **Ranker Agent** - Scores candidates by binding affinity
- **Validator Agent** - Checks safety and novelty constraints
- Full pipeline orchestration with evidence chains

### 🔗 Evidence Chain Visualization
- Trace discovery paths from query to results
- Link compounds to source experiments and publications
- "Explore from Here" navigation for deep dives

### 🎨 Advanced Filtering
- Faceted search with AND/OR logic
- Filter by outcome, cell line, experiment type
- Range sliders for numeric properties

### 🖼️ OCSR (Optical Chemical Structure Recognition)
- Extract SMILES from chemical structure images
- OCR text extraction for formula/name detection
- Support for 2D diagrams and textbook images

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BioFlow Platform                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────────┐    ┌────────────────────┐    ┌──────────────────┐  │
│   │   Next.js 16 UI    │    │   FastAPI Backend  │    │   Qdrant Cloud   │  │
│   │   (React 19 +      │◄──►│   (Python 3.10+)   │◄──►│   Vector DB      │  │
│   │   TypeScript 5.9)  │    │                    │    │   (23K+ vectors) │  │
│   └────────────────────┘    └────────────────────┘    └──────────────────┘  │
│            │                         │                                       │
│            ▼                         ▼                                       │
│   ┌────────────────────┐    ┌────────────────────────────────────────────┐  │
│   │   Dashboard Pages  │    │          AI/ML Components                   │  │
│   │   ├─ Discovery     │    │   ┌──────────┐  ┌───────────┐              │  │
│   │   ├─ Explorer      │    │   │DeepPurpose│  │ ChemBERTa │              │  │
│   │   ├─ Molecules 2D  │    │   │(DTI Pred) │  │(Mol Embed)│              │  │
│   │   ├─ Molecules 3D  │    │   └──────────┘  └───────────┘              │  │
│   │   ├─ Proteins 3D   │    │   ┌──────────┐  ┌───────────┐              │  │
│   │   ├─ Visualization │    │   │  ESM-2   │  │ PubMedBERT│              │  │
│   │   ├─ Data Browser  │    │   │(Prot Emb)│  │(Text Embed)│              │  │
│   │   └─ Workflow      │    │   └──────────┘  └───────────┘              │  │
│   └────────────────────┘    │   ┌──────────────────────────┐             │  │
│                             │   │    Agent Pipeline         │             │  │
│   ┌────────────────────┐    │   │  Generator → Ranker →    │             │  │
│   │   UI Components    │    │   │  Validator (w/ MMR)      │             │  │
│   │   ├─ Evidence Chain│    │   └──────────────────────────┘             │  │
│   │   ├─ Explore Menu  │    └────────────────────────────────────────────┘  │
│   │   ├─ Variant Card  │                                                    │
│   │   └─ Faceted Filter│                                                    │
│   └────────────────────┘                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Multimodal Embedding Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    OpenBioMed Embedding Layer                            │
│                                                                          │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│   │  PubMedBERT  │   │  ChemBERTa   │   │    ESM-2     │                │
│   │   (Text)     │   │  (Molecules) │   │  (Proteins)  │                │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                │
│          │                  │                   │                        │
│          └──────────────────┼───────────────────┘                        │
│                             ▼                                            │
│              ┌──────────────────────────┐                                │
│              │  Unified 768-dim Vectors │                                │
│              └──────────────┬───────────┘                                │
│                             ▼                                            │
│              ┌──────────────────────────┐                                │
│              │   Qdrant HNSW Index      │                                │
│              │   + Payload Metadata     │                                │
│              └──────────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Model Performance

### DTI Prediction Benchmarks

| Dataset | Concordance Index | Pearson Correlation | MSE |
|---------|:-----------------:|:-------------------:|:---:|
| **DAVIS** | 0.7914 | 0.5446 | 0.4684 |
| **KIBA** | 0.7003 | 0.5219 | 0.0008 |
| **BindingDB_Kd** | 0.8083 | 0.7679 | 0.6668 |

### Embedding Models

| Model | Modality | Dimension | Use Case |
|-------|----------|:---------:|----------|
| ChemBERTa | Molecule | 768 | SMILES embeddings |
| ESM-2 | Protein | 768 | Sequence embeddings |
| PubMedBERT | Text | 768 | Biomedical literature |
| BiomedCLIP | Image | 512 | Microscopy, gels, spectra |
| DeepPurpose | DTI | - | Binding affinity prediction |

---

## 🛠️ Technology Stack

### Backend
- **Python 3.10+** - Core runtime
- **FastAPI 0.115** - High-performance async API
- **Uvicorn** - ASGI server
- **PyTorch** - Deep learning framework
- **RDKit** - Cheminformatics toolkit
- **Qdrant Client** - Vector database operations
- **DeepPurpose** - DTI prediction library
- **Transformers** - HuggingFace model hub

### Frontend
- **Next.js 16** - React framework with App Router
- **React 19** - UI library (latest)
- **TypeScript 5.9** - Type-safe JavaScript
- **Tailwind CSS 4** - Utility-first styling
- **Radix UI** - Headless component primitives
- **shadcn/ui** - Beautiful component library
- **3Dmol.js** - Molecular visualization
- **Recharts** - Data visualization
- **Framer Motion** - Animations

### Infrastructure
- **Qdrant Cloud** - Managed vector database
- **Hugging Face Spaces** - Backend deployment
- **Vercel** - Frontend deployment (optional)
- **Docker** - Containerization

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- Node.js 18+ with pnpm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/hamzasammoud11-dotcom/lacoste001.git
cd lacoste001
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Set Up Frontend

```bash
cd ui
pnpm install
cd ..
```

### 4. Configure Environment

Create a `.env` file in the project root:

```env
# Qdrant Cloud (Required)
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-api-key

# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
```

> **Note**: Get free Qdrant Cloud credentials at [cloud.qdrant.io](https://cloud.qdrant.io)

---

## 🚀 Quick Start

### Option 1: Start Both Servers

```bash
# Terminal 1 - Backend
python -m uvicorn bioflow.api.server:app --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd ui
pnpm dev
```

### Option 2: Using start_server.py

```bash
python start_server.py
```

### Access Points

| Service | URL |
|---------|-----|
| **Frontend** | http://localhost:3000 |
| **API Docs** | http://localhost:8000/docs |
| **API Health** | http://localhost:8000/health |

---

## 📂 Project Structure

```
lacoste001/
├── README.md                 # This file
├── BIOFLOW_README.md         # Technical documentation
├── requirements.txt          # Python dependencies
├── start_server.py           # Quick start script
│
├── bioflow/                  # Core Python Package
│   ├── api/                  # FastAPI Backend
│   │   ├── server.py         # Main API server (5400+ lines)
│   │   ├── dti_predictor.py  # DTI prediction service
│   │   ├── qdrant_service.py # Vector database service
│   │   └── model_service.py  # Unified model access
│   │
│   ├── agents/               # Multi-Agent System
│   │   ├── generator.py      # Compound generation
│   │   ├── ranker.py         # Affinity ranking (MMR)
│   │   ├── validator.py      # Safety validation
│   │   └── workflow.py       # Pipeline orchestration
│   │
│   ├── core/                 # Core Abstractions
│   │   ├── config.py         # Configuration management
│   │   ├── nodes.py          # Pipeline nodes
│   │   └── orchestrator.py   # Workflow orchestration
│   │
│   ├── plugins/              # Model Integrations
│   │   ├── deeppurpose_predictor.py
│   │   ├── obm_encoder.py    # OpenBioMed encoders
│   │   ├── qdrant_retriever.py
│   │   └── encoders/         # OCSR, image encoders
│   │
│   ├── search/               # Search Algorithms
│   │   └── enhanced_search.py # MMR, filtering
│   │
│   └── ingestion/            # Data Pipelines
│       ├── chembl_ingestor.py
│       ├── pubmed_ingestor.py
│       ├── uniprot_ingestor.py
│       └── image_ingestor.py
│
├── ui/                       # Next.js 16 Frontend
│   ├── app/
│   │   ├── dashboard/
│   │   │   ├── discovery/    # Drug discovery interface
│   │   │   ├── explorer/     # 3D embedding visualization
│   │   │   ├── molecules-2d/ # SMILES 2D viewer
│   │   │   ├── molecules-3d/ # 3Dmol.js 3D viewer
│   │   │   ├── proteins-3d/  # Protein structure viewer
│   │   │   ├── visualization/# Embedding space explorer
│   │   │   ├── data/         # Data browser
│   │   │   ├── workflow/     # Visual pipeline builder
│   │   │   └── settings/     # Configuration
│   │   └── api/              # Next.js API routes
│   │
│   ├── components/
│   │   ├── dashboard/        # Dashboard-specific components
│   │   │   ├── evidence-chain.tsx      # Evidence visualization
│   │   │   ├── explore-from-here.tsx   # Navigation dropdown
│   │   │   ├── enhanced-filters.tsx    # Faceted filtering
│   │   │   └── variant-justification.tsx # Variant cards
│   │   ├── ui/               # shadcn/ui components
│   │   └── visualization/    # 3D viewers
│   │
│   ├── lib/                  # Utilities & services
│   │   ├── api.ts            # API client
│   │   └── mock-data.ts      # Development data
│   │
│   └── schemas/              # TypeScript types
│
└── data/                     # Datasets
    ├── davis.tab             # DAVIS benchmark
    ├── kiba.tab              # KIBA benchmark
    └── images/               # Indexed images
```

---

## 📡 API Reference

### Health & Status

| Endpoint | Method | Description |
|----------|:------:|-------------|
| `/health` | GET | Service health + model status |
| `/api/status` | GET | Detailed system status |

### Search & Discovery

| Endpoint | Method | Description |
|----------|:------:|-------------|
| `/api/search` | POST | Similarity search by SMILES/text |
| `/api/points` | GET | 3D PCA embedding coordinates |
| `/api/molecules` | GET | Browse indexed molecules |
| `/api/proteins` | GET | Browse indexed proteins |

### Prediction

| Endpoint | Method | Description |
|----------|:------:|-------------|
| `/api/predict` | POST | DTI binding prediction |
| `/api/encode` | POST | Encode molecule/protein/text |
| `/api/validate-smiles` | POST | Validate SMILES strings |

### Agent Pipeline

| Endpoint | Method | Description |
|----------|:------:|-------------|
| `/api/agents/generate` | POST | Generate candidate molecules |
| `/api/agents/rank` | POST | Rank by binding affinity |
| `/api/agents/validate` | POST | Validate safety/novelty |
| `/api/agents/workflow` | POST | Run full agent pipeline |

### Exploration

| Endpoint | Method | Description |
|----------|:------:|-------------|
| `/api/explore/{id}` | POST | Get related items for exploration |
| `/api/evidence-chain/{id}` | GET | Get evidence chain for result |

### Example: Search Similar Compounds

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Nc1ccc(O)cc1", "top_k": 10}'
```

### Example: Predict DTI

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CC(=O)Nc1ccc(O)cc1",
    "target_sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ..."
  }'
```

---

## 🖥️ Dashboard Modules

### 1. Discovery (`/dashboard/discovery`)
The main drug discovery interface featuring:
- Natural language, SMILES, or FASTA input
- Real-time similarity search
- Result cards with scores and metadata
- "Explore from Here" navigation
- Evidence chain visualization

### 2. 3D Explorer (`/dashboard/explorer`)
Interactive 3D embedding space visualization:
- PCA-projected chemical space
- Color-coded by affinity/modality
- Click-to-inspect compounds
- Cluster analysis tools

### 3. Molecules 2D (`/dashboard/molecules-2d`)
2D chemical structure viewer:
- SMILES input rendering
- Property calculation (MW, LogP, TPSA)
- Export capabilities

### 4. Molecules 3D (`/dashboard/molecules-3d`)
3D molecular visualization:
- Interactive 3Dmol.js viewer
- Multiple rendering styles
- Surface visualization

### 5. Proteins 3D (`/dashboard/proteins-3d`)
Protein structure viewer:
- PDB structure loading
- AlphaFold integration
- Sequence highlighting

### 6. Visualization (`/dashboard/visualization`)
Embedding space analysis:
- UMAP/t-SNE projections
- Modality filtering
- Similarity clustering

### 7. Data Browser (`/dashboard/data`)
Browse indexed data:
- Experiments listing
- Compound database
- Protein targets

### 8. Workflow Builder (`/dashboard/workflow`)
Visual pipeline construction:
- Drag-and-drop nodes
- Custom pipeline creation
- Export configurations

---

## ☁️ Deployment

### Backend: Hugging Face Spaces

The FastAPI backend is deployed on HF Spaces with Docker:

```dockerfile
# Dockerfile.hf
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "bioflow.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Frontend: Vercel

```bash
cd ui
vercel --prod
```

### Environment Variables

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

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 Resources

- [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose) - DTI Prediction Toolkit
- [OpenBioMed](https://github.com/PharMolix/OpenBioMed) - Multimodal Bio-AI
- [Qdrant](https://qdrant.tech/) - Vector Database
- [3Dmol.js](https://3dmol.csb.pitt.edu/) - Molecular Visualization
- [shadcn/ui](https://ui.shadcn.com/) - UI Components

---

## 👥 Team Lacoste

Built with ❤️ for biological discovery.

| Role | Focus |
|------|-------|
| Platform Architecture | System design & integration |
| AI/ML Engineering | Model training & deployment |
| Frontend Development | React/Next.js UI |
| Backend Development | FastAPI services |

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>BioFlow</b> - Accelerating Drug Discovery with AI
  <br>
  <sub>© 2024-2026 Team Lacoste</sub>
</p>
