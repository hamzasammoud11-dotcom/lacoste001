# BioFlow: AI-Powered Drug-Target Interaction Platform
## Technical Report - January 2026

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Core Technologies](#core-technologies)
4. [Pipeline Implementation](#pipeline-implementation)
5. [Model Training & Results](#model-training--results)
6. [Qdrant Vector Database Integration](#qdrant-vector-database-integration)
7. [FastAPI Backend](#fastapi-backend)
8. [Frontend Application](#frontend-application)
9. [Langflow Integration](#langflow-integration)
10. [Current Status](#current-status)
11. [Future Roadmap](#future-roadmap)

---

## Executive Summary

**BioFlow** is an end-to-end AI-powered drug discovery platform designed for predicting Drug-Target Interactions (DTI). The system combines deep learning models (DeepPurpose), vector similarity search (Qdrant), and a modern React-based frontend to enable researchers to:

- Train and evaluate DTI prediction models on benchmark datasets
- Perform similarity search across drug-target embedding space
- Visualize molecular structures in 2D and 3D
- Build visual pipelines using Langflow for no-code experimentation

### Key Achievements
- ✅ Trained models on **3 benchmark datasets** (KIBA, DAVIS, BindingDB_Kd)
- ✅ Best Concordance Index (CI): **0.805** on BindingDB_Kd
- ✅ Indexed **23,531 drug-target pairs** in Qdrant vector database
- ✅ Real-time similarity search via FastAPI backend
- ✅ Interactive 2D/3D molecular visualization
- ✅ Langflow pipeline for no-code DTI prediction

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BioFlow Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐   │
│   │   Frontend   │────▶│  FastAPI     │────▶│      Qdrant              │   │
│   │   (Next.js)  │     │  Backend     │     │   Vector Database        │   │
│   │   Port 3000  │     │  Port 8001   │     │   Port 6333              │   │
│   └──────────────┘     └──────────────┘     └──────────────────────────┘   │
│          │                    │                        ▲                    │
│          │                    │                        │                    │
│          ▼                    ▼                        │                    │
│   ┌──────────────┐     ┌──────────────┐     ┌─────────┴──────────────┐     │
│   │  3Dmol.js    │     │ DeepPurpose  │     │   Ingestion Pipeline   │     │
│   │  Smiles-     │     │   Model      │     │   (ingest_qdrant.py)   │     │
│   │  Drawer      │     │   (PyTorch)  │     └────────────────────────┘     │
│   └──────────────┘     └──────────────┘                                    │
│                               │                                             │
│                               ▼                                             │
│                        ┌──────────────┐     ┌────────────────────────┐     │
│                        │   Langflow   │────▶│   Visual Pipeline      │     │
│                        │   Port 7860  │     │   (DTI Orchestrator)   │     │
│                        └──────────────┘     └────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Technologies

### Backend Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| ML Framework | **DeepPurpose** (PyTorch) | Drug-Target Interaction prediction |
| Vector DB | **Qdrant** | Similarity search on embeddings |
| API Server | **FastAPI** + Uvicorn | REST API for frontend |
| Data Source | **TDC (Therapeutics Data Commons)** | Benchmark DTI datasets |
| Pipeline UI | **Langflow** | No-code visual pipeline builder |

### Frontend Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | **Next.js 16** (App Router) | React server components |
| UI Library | **Shadcn/UI** + Radix | Component system |
| Styling | **Tailwind CSS** | Utility-first CSS |
| 3D Viz | **3Dmol.js** | Protein structure viewer |
| 2D Viz | **smiles-drawer** | Molecule structure rendering |
| Package Manager | **pnpm** | Fast, disk-efficient |

---

## Pipeline Implementation

### 1. Training Pipeline (`deeppurpose002.py`)

The training script is a comprehensive CLI tool that:

```bash
python deeppurpose002.py --dataset KIBA --epochs 10 --drug_enc Morgan --target_enc CNN
```

**Features:**
- Automatic GPU detection (CUDA support)
- Multiple dataset support: DAVIS, KIBA, BindingDB_Kd, BindingDB_Ki, BindingDB_IC50
- Label transformation: `paffinity_nm` (converts nM to -log10 scale)
- Comprehensive metrics: MSE, RMSE, MAE, Pearson, Spearman, Concordance Index
- Automatic visualization generation (scatter plots, residuals, sorted curves)

**Encoding Configuration:**
```python
MODEL_CONFIG = {
    "drug_encoding": "Morgan",      # Morgan fingerprints (1024-bit)
    "target_encoding": "CNN",       # CNN for protein sequences
    "cls_hidden_dims": [1024, 1024, 512],
    "hidden_dim_drug": 128,
    "hidden_dim_protein": 128,
}
```

### 2. Ingestion Pipeline (`ingest_qdrant.py`)

Converts trained model embeddings into searchable vectors:

```
[1/6] Load Model (model.pt + config.pkl)
[2/6] Load Dataset from TDC (KIBA test split)
[3/6] Generate Embeddings (no shuffle to preserve order)
[4/6] Compute PCA projections (drug, target, combined)
[5/6] Connect to Qdrant (localhost:6333)
[6/6] Upload points with payloads
```

**Vector Schema:**
```python
vectors_config = {
    "drug": VectorParams(size=128, distance=Distance.COSINE),
    "target": VectorParams(size=128, distance=Distance.COSINE),
}
```

**Payload Structure:**
```json
{
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "target_seq": "MKTAYIAK...",
  "label_true": 7.2,
  "pca_drug": [0.12, -0.34, 0.56],
  "pca_target": [-0.21, 0.78, 0.11],
  "pca_combined": [0.45, -0.12, 0.67],
  "affinity_class": "high"  // high: >7, medium: 5-7, low: <5
}
```

---

## Model Training & Results

### Benchmark Performance

| Dataset | Samples | CI | Pearson | MSE | Training Time |
|---------|---------|-----|---------|-----|---------------|
| **BindingDB_Kd** | 42,227 | **0.805** | 0.768 | 0.667 | 1h 49m |
| **KIBA** | 117,656 | 0.703 | 0.522 | 0.0008 | 3h 42m |
| **DAVIS** | 25,772 | 0.786 | 0.545 | 0.468 | 9m |

### Best Model Configuration
- **Selected Run:** `20260125_104915_KIBA`
- **Hardware:** NVIDIA GeForce RTX 3070 Laptop GPU
- **Epochs:** 10
- **Batch Size:** 256
- **Learning Rate:** 1e-4
- **Split:** 80/10/10 (train/val/test)

### Metrics Explanation
- **Concordance Index (CI):** Probability that predictions preserve true ordering (0.5 = random, 1.0 = perfect)
- **Pearson Correlation:** Linear correlation between true and predicted values
- **MSE:** Mean Squared Error (lower is better)

---

## Qdrant Vector Database Integration

### Collection: `bio_discovery`

**Statistics:**
- Total Vectors: **23,531** drug-target pairs
- Vector Dimensions: 128 (drug) + 128 (target)
- Distance Metric: Cosine Similarity
- Pre-computed PCA: 3D projections for visualization

### Search Capabilities

**1. Drug Similarity Search**
```python
# Input: SMILES string
query = "CC(=O)Nc1ccc(O)cc1"  # Acetaminophen
# Output: Top-K similar drugs by Morgan fingerprint embedding
```

**2. Target Similarity Search**
```python
# Input: Protein sequence
query = "MKTAYIAKQRQISFVKSHFSRQLE..."
# Output: Top-K similar targets by CNN embedding
```

**3. Text Search (Fallback)**
```python
# Input: Partial SMILES or keyword
# Output: Substring matches in payload
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | POST | Vector similarity search |
| `/api/points` | GET | Get points for 3D visualization |
| `/api/stats` | GET | Collection statistics |
| `/health` | GET | Service health check |

---

## FastAPI Backend

### Server: `server/api.py`

**Startup Sequence:**
```
[STARTUP] Loading DeepPurpose model...
[STARTUP] Model loaded from runs\20260125_104915_KIBA\model.pt
[STARTUP] Using device: cuda
[STARTUP] Connecting to Qdrant...
[STARTUP] Connected. Collections: ['bio_discovery']
[STARTUP] Ready!
```

**Key Features:**
1. **Model Caching:** Model loaded once at startup (not per-request)
2. **Device Override:** Fixes DeepPurpose's global device variable for GPU inference
3. **CORS Enabled:** Allows frontend on port 3000
4. **Error Handling:** Fallback to text search if encoding fails

### Direct Encoding (No data_process)

The API uses direct encoding to avoid DeepPurpose's `data_process` overhead:

```python
# Drug encoding (Morgan fingerprints)
from DeepPurpose.utils import smiles2morgan
morgan_fp = smiles2morgan(smiles, radius=2, nBits=1024)
vector = model.model.model_drug(torch.tensor([morgan_fp]))

# Target encoding (CNN)
from DeepPurpose.utils import trans_protein
target_encoding = trans_protein(sequence)
vector = model.model.model_protein(torch.tensor([target_encoding]))
```

---

## Frontend Application

### Page Structure

```
ui/app/
├── page.tsx                 # Landing page
├── layout.tsx               # Root layout (ThemeProvider)
└── dashboard/
    ├── page.tsx             # Dashboard home
    ├── discovery/           # Drug discovery search
    │   └── page.tsx
    ├── explorer/            # Data exploration
    │   ├── page.tsx
    │   ├── chart.tsx
    │   └── components.tsx
    ├── molecules-2d/        # 2D molecule viewer
    │   ├── page.tsx
    │   └── _components/
    │       └── Smiles2DViewer.tsx
    ├── molecules-3d/        # 3D molecule viewer
    │   ├── page.tsx
    │   └── _components/
    │       └── Molecule3DViewer.tsx
    └── proteins-3d/         # 3D protein viewer
        ├── page.tsx
        └── _components/
            └── ProteinViewer.tsx
```

### Key Components

**1. Discovery Page**
- Input: SMILES or protein sequence
- Search types: Similarity, Text
- Results: Ranked list with affinity scores

**2. Molecules 2D Viewer**
- Renders molecules using `smiles-drawer`
- Supports common molecules (Caffeine, Aspirin, etc.)
- Copy SMILES functionality

**3. Proteins 3D Viewer**
- Uses `3Dmol.js` for WebGL rendering
- Fetches PDB files from RCSB
- Multiple representation styles (cartoon, surface, stick)

**4. Explorer**
- 3D scatter plot of embedding space
- Color-coded by affinity class
- Interactive point selection

---

## Langflow Integration

### Purpose

Langflow provides a **no-code visual interface** for building DTI prediction pipelines. It allows researchers without coding experience to:

1. Create drug-target interaction workflows
2. Chain API calls visually
3. Filter results based on affinity thresholds
4. Export predictions

### Pipeline Configuration

File: `langflow/bioflow_dti_pipeline.json`

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Drug Input     │────▶│   DeepPurpose   │────▶│    Qdrant       │
│  (SMILES)       │     │   Encoder       │     │  Vector Store   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
┌─────────────────┐                                     ▼
│  Target Input   │────▶                        ┌─────────────────┐
│  (Protein Seq)  │                             │  Affinity       │
└─────────────────┘                             │  Filter (>0.8)  │
                                                └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │  DTI Results    │
                                                │  Output         │
                                                └─────────────────┘
```

### Running Langflow

```bash
# Start Langflow server
.\.venv\Scripts\langflow run --host 0.0.0.0 --port 7860

# Or use the dedicated venv
.\langflow_venv\Scripts\langflow run --host 0.0.0.0 --port 7860
```

**Note:** Langflow requires a separate virtual environment due to dependency conflicts with DeepPurpose.

---

## Current Status

### ✅ Completed Features

| Feature | Status | Notes |
|---------|--------|-------|
| DeepPurpose Training Pipeline | ✅ Done | 3 datasets trained |
| Qdrant Ingestion | ✅ Done | 23,531 vectors indexed |
| FastAPI Backend | ✅ Done | Running on port 8001 |
| Vector Search API | ✅ Done | Drug/Target similarity |
| Next.js Frontend | ✅ Done | 6 pages implemented |
| 2D Molecule Viewer | ✅ Done | smiles-drawer integration |
| 3D Protein Viewer | ✅ Done | 3Dmol.js integration |
| Dark Mode | ✅ Done | next-themes provider |
| Langflow Pipeline | ✅ Done | JSON config ready |

### 🚧 Partially Complete

| Feature | Status | Notes |
|---------|--------|-------|
| 3D Molecule Viewer | 🚧 WIP | SDF fetching needs work |
| Explorer Visualization | 🚧 WIP | Chart rendering issues |
| Data Page | 🚧 WIP | API stats integration |

### ❌ Not Yet Implemented

| Feature | Priority | Description |
|---------|----------|-------------|
| OpenBioMed Integration | High | Multi-modal foundation model |
| User Authentication | Medium | Login/session management |
| Batch Predictions | Medium | Upload CSV for bulk inference |
| Model Fine-tuning UI | Low | Retrain on custom data |
| Export Results | Low | CSV/JSON download |

---

## Future Roadmap

### Phase 1: OpenBioMed Integration (High Priority)

[OpenBioMed](https://github.com/PharMolix/OpenBioMed) is a multi-modal foundation model for biomedicine that would significantly enhance BioFlow's capabilities:

**Planned Features:**
1. **Molecule-Text Alignment**
   - Search drugs using natural language descriptions
   - Example: "Find molecules similar to aspirin that reduce inflammation"

2. **Protein-Text Alignment**
   - Describe targets in plain English
   - Example: "Kinase involved in cancer cell proliferation"

3. **Cross-Modal Retrieval**
   - Find drugs for a given text description
   - Find targets for a given drug structure

4. **Enhanced Embeddings**
   - Replace Morgan/CNN with transformer-based encoders
   - Better generalization to novel compounds

**Implementation Plan:**
```python
# Replace current encoding
# FROM: Morgan fingerprints + CNN
# TO: OpenBioMed's BioMedGPT encoder

from openbiomedgpt import BioMedGPTEncoder
encoder = BioMedGPTEncoder.load_pretrained("biomedgpt-base")

# Multi-modal embedding
drug_embedding = encoder.encode_molecule(smiles)
target_embedding = encoder.encode_protein(sequence)
text_embedding = encoder.encode_text("kinase inhibitor")
```

### Phase 2: Advanced Search & Filtering

1. **Faceted Search**
   - Filter by molecular weight, logP, TPSA
   - Filter by target family (kinases, GPCRs, etc.)

2. **ADMET Predictions**
   - Absorption, Distribution, Metabolism, Excretion, Toxicity
   - Integrate with ADMETlab 2.0

3. **Structure-Activity Relationship (SAR)**
   - Identify key structural features
   - Scaffold hopping suggestions

### Phase 3: Collaboration Features

1. **Project Workspaces**
   - Save searches and results
   - Share with team members

2. **Annotation System**
   - Tag molecules with notes
   - Track experimental validation

3. **Integration with Lab Notebooks**
   - Export to ELN systems
   - Import experimental data

---

## Running the System

### Quick Start

```powershell
# 1. Start Qdrant (Docker)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 2. Start Backend API
.\.venv\Scripts\python -m uvicorn server.api:app --host 0.0.0.0 --port 8001

# 3. Start Frontend
cd ui
pnpm dev

# 4. (Optional) Start Langflow
.\langflow_venv\Scripts\langflow run --host 0.0.0.0 --port 7860
```

### Service Ports
| Service | Port | URL |
|---------|------|-----|
| Frontend | 3000 | http://localhost:3000 |
| Backend API | 8001 | http://localhost:8001 |
| Qdrant | 6333 | http://localhost:6333 |
| Langflow | 7860 | http://localhost:7860 |

### Health Check
```powershell
# Check all services
$qdrant = netstat -ano | Select-String ":6333.*LISTENING"
$api = netstat -ano | Select-String ":8001.*LISTENING"
$ui = netstat -ano | Select-String ":3000.*LISTENING"

Write-Host "Qdrant: $($qdrant -ne $null)"
Write-Host "API: $($api -ne $null)"
Write-Host "UI: $($ui -ne $null)"
```

---

## File Structure

```
lacoste001/
├── config.py                    # Shared configuration
├── deeppurpose002.py            # Training pipeline
├── ingest_qdrant.py             # Vector ingestion
├── runs/                        # Model checkpoints & results
│   ├── 20260125_080409_BindingDB_Kd/
│   ├── 20260125_104915_KIBA/    # Best model ★
│   └── 20260126_160009_DAVIS/
├── server/
│   └── api.py                   # FastAPI backend
├── langflow/
│   └── bioflow_dti_pipeline.json
├── data/
│   ├── davis.tab
│   └── kiba.tab
└── ui/                          # Next.js frontend
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx
    │   └── dashboard/
    │       ├── discovery/
    │       ├── explorer/
    │       ├── molecules-2d/
    │       ├── molecules-3d/
    │       └── proteins-3d/
    ├── components/
    └── lib/
```

---

## Conclusion

BioFlow demonstrates a complete pipeline for AI-powered drug discovery, from model training to interactive visualization. The system successfully:

1. **Trains** DTI prediction models achieving CI > 0.80
2. **Indexes** embeddings for fast similarity search
3. **Serves** predictions via REST API
4. **Visualizes** molecules and proteins in the browser
5. **Enables** no-code experimentation via Langflow

The next major milestone is **OpenBioMed integration**, which will unlock multi-modal search and dramatically improve the user experience for drug discovery researchers.

---

*Report generated: January 26, 2026*  
*Repository: github.com/hamzasammoud11-dotcom/lacoste001*  
*Branch: core-progress*
