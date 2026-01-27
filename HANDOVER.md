# AI Handover Document - BioFlow Project

## 1. Project Context & Strategic Vision

### Project Name

**BioFlow** (part of the OpenBioMed ecosystem)

### Mission Statement

Build a **fully open-source** AI-powered biological discovery platform that unifies fragmented R&D data (text, sequences, molecules, structures) into an intelligent exploration and design engine.

### Target Use Case (UC4)

**"Multimodal Biological Design & Discovery Intelligence"**

- Ingest and index multimodal biological data
- Enable cross-modal similarity search (text ↔ molecule ↔ protein)
- Suggest "close but diverse" variants for design exploration
- Provide scientific evidence linking and traceability

### ⚠️ Critical Constraint

**This project must be 100% open-source.** InstaDeep models (ProtBFN, AbBFN2, Laila, DeepChain, InstaNovo) are proprietary and **cannot** be used. All models must come from open repositories (HuggingFace, GitHub, etc.).

---

## 2. Where Does OBM (OpenBioMed) Stand?

### OBM's Role in the Architecture

OBM is the **multimodal embedding backbone** - it is ONE tool among several in the BioFlow platform:

```
┌────────────────────────────────────────────────────────────────────────┐
│                         BioFlow Platform                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    OBM (Embedding Layer)                          │  │
│  │  ┌──────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │ PubMedBERT│  │  ChemBERTa   │  │    ESM-2     │                │  │
│  │  │  (Text)   │  │  (Molecules) │  │  (Proteins)  │                │  │
│  │  └─────┬─────┘  └──────┬───────┘  └──────┬───────┘                │  │
│  │        └───────────────┼─────────────────┘                        │  │
│  │                        ▼                                          │  │
│  │              Unified 768-dim Embeddings                           │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │                                          │
│  ┌──────────────────────────▼───────────────────────────────────────┐  │
│  │                    Qdrant (Vector Memory)                         │  │
│  │  • HNSW indexing for fast similarity search                       │  │
│  │  • Payload storage (metadata, source, tags)                       │  │
│  │  • Filtered retrieval by modality/source                          │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │                                          │
│  ┌──────────────────────────▼───────────────────────────────────────┐  │
│  │                    Agent Pipeline                                 │  │
│  │  ┌─────────┐  ┌───────────┐  ┌─────────┐  ┌─────────┐            │  │
│  │  │ Miner   │  │ Generator │  │Validator│  │ Ranker  │            │  │
│  │  │(Lit.)   │  │(DeepPurpose│  │(Toxicity│  │(MMR/Div)│            │  │
│  │  └─────────┘  │ MolT5)     │  │ KG)     │  └─────────┘            │  │
│  │               └───────────┘  └─────────┘                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Next.js UI                                     │  │
│  │  • Search interface (text/SMILES/sequence)                        │  │
│  │  • 3D visualization (embedding space)                             │  │
│  │  • Evidence linking & traceability                                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### What OBM Provides

| Component | Model | Source | Purpose |
|-----------|-------|--------|---------|
| TextEncoder | PubMedBERT | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | Encode biomedical abstracts |
| MoleculeEncoder | ChemBERTa | `seyonec/ChemBERTa-zinc-base-v1` | Encode SMILES molecules |
| ProteinEncoder | ESM-2 | `facebook/esm2_t12_35M_UR50D` | Encode protein sequences |

### What OBM Does NOT Provide

- **Generation**: Molecule/protein design (use DeepPurpose, MolT5, ESMFold)
- **Validation**: Toxicity/ADMET prediction (use external KGs or predictors)
- **Orchestration**: Workflow management (use BioFlow pipeline)

---

## 3. Full Architecture & Components

### Layer 1: Data Ingestion

| Data Source | Type | Format |
|-------------|------|--------|
| PubMed | Text | Abstracts (JSON/XML) |
| UniProt | Protein | FASTA sequences |
| ChEMBL | Molecule | SMILES strings |
| BioImage Archive | Image | (Future: CLIP embeddings) |

### Layer 2: Embedding (OBM)

- **bioflow/plugins/obm_encoder.py**: Central encoder class
- **bioflow/plugins/encoders/**: Modality-specific encoders
  - `text_encoder.py` → PubMedBERT
  - `molecule_encoder.py` → ChemBERTa / RDKit fingerprints
  - `protein_encoder.py` → ESM-2
- **bioflow/obm_wrapper.py**: High-level API (`encode_text`, `encode_smiles`, `encode_protein`)

### Layer 3: Vector Storage (Qdrant)

- **bioflow/qdrant_manager.py**: Low-level Qdrant operations
- **bioflow/api/qdrant_service.py**: API-level service with caching
- Collections: `molecules`, `proteins`, `texts` (or unified `bioflow_memory`)

### Layer 4: Agent Pipeline

- **bioflow/pipeline.py**: Workflow orchestration
  - `MinerAgent`: Literature retrieval
  - `ValidatorAgent`: Safety/toxicity checks
  - `RankerAgent`: MMR-based diversification
- **bioflow/api/deeppurpose_api.py**: DTI prediction endpoints

### Layer 5: API Server

- **bioflow/api/server.py**: FastAPI application
- Endpoints: `/health`, `/api/molecules`, `/api/proteins`, `/api/search`, `/api/points`, `/api/collections`

### Layer 6: Frontend

- **ui/**: Next.js 16 application
- Pages: Discovery, Explorer, Molecules, Proteins
- Mock fallbacks in `ui/app/api/_mock/`

---

## 4. Current Implementation Status

### ✅ Implemented

| Component | Status | Notes |
|-----------|--------|-------|
| OBMEncoder | ✅ Working | PubMedBERT + ChemBERTa + ESM-2 |
| Qdrant Integration | ✅ Working | Local storage at `./qdrant_data` |
| FastAPI Server | ✅ Working | Port 8000 |
| Next.js UI | ✅ Working | Port 3000 |
| DeepPurpose Integration | ⚠️ Optional | Requires `DeepPurpose` package |
| Mock Fallbacks | ✅ Implemented | For offline/demo mode |

### ❌ Not Yet Implemented

| Feature | Priority | Notes |
|---------|----------|-------|
| Image Embeddings | Medium | Need BioMedCLIP or similar |
| MMR Diversification | High | In RankerAgent (basic) |
| Evidence Linking | High | Need to add source tracking |
| PubMed Ingestion | High | Need data pipeline |
| UniProt Ingestion | High | Need data pipeline |
| ChEMBL Ingestion | Medium | Need data pipeline |
| 3D Visualization | Medium | `/api/points` endpoint ready |

---

## 5. How to Run

### Prerequisites

```bash
# Python 3.9+
pip install torch transformers qdrant-client fastapi uvicorn

# Optional (for full functionality)
pip install rdkit-pypi DeepPurpose

# Node.js 18+
npm install -g pnpm
```

### Start Backend

```bash
cd c:\Users\ramit\OneDrive\Bureau\Github\OpenBioMed
python -m uvicorn bioflow.api.server:app --host 0.0.0.0 --port 8000
```

### Start Frontend

```bash
cd ui
pnpm install
pnpm dev
```

---

## 6. Open-Source Models Used

| Task | Model | License | HuggingFace Path |
|------|-------|---------|------------------|
| Text Embedding | PubMedBERT | MIT | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` |
| Text Embedding | SciBERT | Apache 2.0 | `allenai/scibert_scivocab_uncased` |
| Molecule Embedding | ChemBERTa | MIT | `seyonec/ChemBERTa-zinc-base-v1` |
| Protein Embedding | ESM-2 | MIT | `facebook/esm2_t12_35M_UR50D` |
| DTI Prediction | DeepPurpose | BSD | [GitHub](https://github.com/kexinhuang12345/DeepPurpose) |
| Molecule Generation | MolT5 | Apache 2.0 | `laituan245/molt5-base` |
| Protein Folding | ESMFold | MIT | `facebook/esmfold_v1` |

**NO PROPRIETARY MODELS FROM INSTADEEP ARE USED.**

---

## 7. Next Steps for AI Assistant

1. **Data Ingestion Pipelines**: Create scripts to ingest PubMed, UniProt, ChEMBL data
2. **Evidence Linking**: Add source tracking to all search results
3. **MMR Diversification**: Implement proper Maximal Marginal Relevance in RankerAgent
4. **Image Support**: Integrate BioMedCLIP for bioimaging data
5. **Evaluation Metrics**: Implement Recall@10, MRR, nDCG for retrieval quality
6. **Batch Processing**: Optimize for large-scale data ingestion
