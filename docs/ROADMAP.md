# 🗺️ BioFlow Orchestrator Development Roadmap

This roadmap outlines the collaborative development of a unified R&D platform for biological discovery using **fully open-source** tools and models.

---

## 🏗️ Phase 1: Infrastructure & Core Framework ✅ COMPLETE
**Goal:** Establish the "modality-agnostic" foundation so tools can be plugged in without rewriting core logic.

- [x] **Core Abstractions** (`bioflow/core/base.py`):
  - `BioEncoder`: Interface for vectorization (ESM-2, ChemBERTa, PubMedBERT, CLIP)
  - `BioPredictor`: Interface for predictions (DeepPurpose, ADMET)
  - `BioGenerator`: Interface for candidate generation
  - `BioRetriever`: Interface for vector DB operations
  - Data containers: `EmbeddingResult`, `PredictionResult`, `RetrievalResult`

- [x] **Tool Registry** (`bioflow/core/registry.py`):
  - Central hub to manage multiple tools
  - Register/unregister by name
  - Default tool fallbacks
  - Utility methods for listing and summary

- [x] **Configuration Schema** (`bioflow/core/config.py`):
  - `NodeConfig`: Single pipeline node definition
  - `WorkflowConfig`: Complete workflow definition
  - `BioFlowConfig`: Master system configuration
  - YAML-compatible dataclasses

- [x] **Stateful Pipeline Engine** (`bioflow/core/orchestrator.py`):
  - `BioFlowOrchestrator`: DAG-based workflow execution
  - Topological sort for dependency resolution
  - `ExecutionContext` for state passing
  - Custom handler support
  - Error handling and traceability

- [x] **Sample Workflows** (`bioflow/workflows/`):
  - `drug_discovery.yaml`: Encode → Retrieve → Predict → Filter
  - `literature_mining.yaml`: Cross-modal literature search

---

## 🧪 Phase 2: Parallel Tool Implementation ✅ COMPLETE
The team works on their respective modules using the core interfaces.

### **1. OBM Integration** ✅
- [x] `OBMEncoder` - Unified multimodal encoder (`bioflow/plugins/obm_encoder.py`)
- [x] `TextEncoder` - PubMedBERT/SciBERT (`bioflow/plugins/encoders/text_encoder.py`)
- [x] `MoleculeEncoder` - ChemBERTa/RDKit (`bioflow/plugins/encoders/molecule_encoder.py`)
- [x] `ProteinEncoder` - ESM-2/ProtBERT (`bioflow/plugins/encoders/protein_encoder.py`)
- [x] Lazy loading for efficient memory usage
- [x] Dimension projection for cross-modal compatibility

### **2. Qdrant Retriever** ✅
- [x] `QdrantRetriever` implements `BioRetriever` interface (`bioflow/plugins/qdrant_retriever.py`)
- [x] HNSW indexing with cosine/euclidean/dot distance
- [x] Payload filtering (species, experiment type, modality)
- [x] Batch ingestion support
- [x] In-memory, local, or remote Qdrant connections

### **3. DeepPurpose Predictor** ✅
- [x] `DeepPurposePredictor` implements `BioPredictor` (`bioflow/plugins/deeppurpose_predictor.py`)
- [x] DTI prediction with Transformer+CNN architecture
- [x] Graceful fallback when DeepPurpose unavailable
- [x] Batch prediction support

---

## 🔗 Phase 3: The Unified Workflow ✅ COMPLETE
**Goal:** Connect the tools into a coherent discovery loop.

- [x] **Typed Node System** (`bioflow/core/nodes.py`):
  - `EncodeNode`: Vectorize inputs via BioEncoder
  - `RetrieveNode`: Query vector DB for similar items
  - `PredictNode`: Run DTI predictions on candidates
  - `IngestNode`: Add new data to vector DB
  - `FilterNode`: Score-based filtering and ranking
  - `TraceabilityNode`: Link results to evidence sources

- [x] **Discovery Pipelines** (`bioflow/workflows/discovery.py`):
  - `DiscoveryPipeline`: Full drug discovery workflow (encode → retrieve → predict → filter → trace)
  - `LiteratureMiningPipeline`: Cross-modal literature search
  - `ProteinDesignPipeline`: Protein homolog discovery
  - Batch ingestion and simple search APIs

- [x] **Data Ingestion Utilities** (`bioflow/workflows/ingestion.py`):
  - JSON/CSV file loaders
  - SMILES/FASTA file parsers
  - Sample data generators for testing

- [x] **Evidence Traceability**:
  - Automatic PubMed/UniProt/PubChem/DrugBank link generation
  - Metadata preservation through pipeline

**Verification:** `python scripts/verify_phase3.py` - All 5 tests pass ✅

---

## 📊 Phase 4: UI/UX & Deployment ✅ COMPLETE
**Goal:** Build an intuitive, modern interface for the BioFlow platform.

- [x] **Next.js Frontend** (`ui/`):
  - Next.js 16 app router + Tailwind + shadcn/ui
  - Dashboard pages: Discovery, 3D Visualization, Workflow Builder
  - `/app/api/*` proxy routes to the FastAPI backend
  - Optional mock fallbacks for molecules/proteins list routes

**Launch:**
- Full stack (Windows): `launch_bioflow_full.bat`
- Manual:
  - Backend: `python -m uvicorn bioflow.api.server:app --host 0.0.0.0 --port 8000`
  - UI: `cd ui && pnpm dev`

---

## 🚀 Phase 5: Open-Source Alignment
- **Strict Open-Source Compliance**: remove proprietary integrations and keep only OSS models/tools.
- **Open Protein/Peptide Options**: integrate open models (e.g., ESM-2 / ProGen2) behind `BioGenerator`.
- **Open Retrieval + Evidence**: improve evidence traceability (PubMed/UniProt/ChEMBL) and evaluation.
