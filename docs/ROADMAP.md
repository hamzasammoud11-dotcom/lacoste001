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

- [x] **Theme & Styling** (`bioflow/ui/config.py`):
  - Glassmorphism design system with dark theme
  - Custom CSS with animations, cards, badges
  - Responsive layout with Inter font family
  - Color palette: Indigo primary, Emerald success, Cyan accent

- [x] **Reusable Components** (`bioflow/ui/components.py`):
  - `hero_section`: Landing hero with stats
  - `metric_card`: Animated metric displays
  - `glass_card`, `feature_cards`: Content cards
  - `pipeline_flow`: Visual pipeline status
  - `binding_affinity_chart`, `scatter_embedding`, `similarity_heatmap`: Charts
  - `molecule_viewer_2d`: RDKit molecule rendering
  - `evidence_card`: Traceability links
  - `chat_message`, `chat_container`: AI chat interface
  - `step_progress`, `notification`: UX helpers

- [x] **Dashboard Home** (`bioflow/ui/pages/home.py`):
  - Hero section with platform branding
  - Key metrics (molecules, proteins, literature, predictions)
  - Quick action cards (Discovery, Explorer, Upload)
  - Feature highlights grid
  - Recent discoveries chart
  - Activity timeline
  - Active pipeline visualization

- [x] **Discovery Page** (`bioflow/ui/pages/discovery.py`):
  - Query input (text, SMILES, FASTA)
  - Target protein selection with common targets
  - Real-time pipeline progress visualization
  - Results with binding affinity chart
  - Top hits with molecule viewer
  - Evidence linking to PubMed/ChEMBL/PubChem
  - Export options (CSV, SMILES, Report)

- [x] **Explorer Page** (`bioflow/ui/pages/explorer.py`):
  - 2D/3D embedding visualization
  - Dimensionality reduction (t-SNE, PCA, UMAP)
  - Modality filtering
  - Cross-modal similarity heatmap
  - Nearest neighbor search
  - Cluster analysis with K-Means/DBSCAN

- [x] **Data Management Page** (`bioflow/ui/pages/data.py`):
  - Collection overview with metrics
  - File upload (CSV, JSON, SMILES, FASTA)
  - Data preview with molecule rendering
  - Batch processing with progress
  - Collection browsing and search
  - Scheduled task management

- [x] **Settings Page** (`bioflow/ui/pages/settings.py`):
  - Qdrant connection configuration
  - Model selection (PubMedBERT, ESM-2, ChemBERTa)
  - Predictor configuration (DeepPurpose)
  - Theme and appearance settings
  - System status monitoring
  - Resource usage (Memory, GPU, Storage)

- [x] **Main App** (`bioflow/ui/app.py`):
  - Navigation sidebar with routing
  - User profile display
  - Quick stats in sidebar

**Launch:** `python launch_ui.py` or `streamlit run bioflow/ui/app.py`

---

## 🚀 Phase 5: Open-Source Alignment
- **Laila Connector**: Allow Laila to query the Qdrant memory.
- **InstaNovo+ Specs**: Add support for peptide sequencing integration.
- **Controlled Generation**: Pilot generation via ProtBFN/AbBFN2.
