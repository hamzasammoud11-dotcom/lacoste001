# BioFlow - AI-Powered Drug Discovery Platform

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**BioFlow** is a unified AI platform for drug discovery, combining molecular encoding, protein analysis, and drug-target interaction prediction in a modern web interface.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Next.js Frontend                          │
│                   (React 19 + Tailwind)                      │
│                     localhost:3000                           │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP/REST
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                            │
│                    localhost:8000                            │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │ ModelService │  │QdrantService│  │ DTI Predictor    │   │
│  │ (Encoders)   │  │ (VectorDB)  │  │ (DeepPurpose)    │   │
│  └──────────────┘  └─────────────┘  └──────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    OpenBioMed Core                           │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────────┐   │
│  │   Models   │  │  Datasets  │  │       Tasks         │   │
│  │ BioT5,ESM  │  │ DAVIS,KIBA │  │ Property Prediction │   │
│  └────────────┘  └────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ with pnpm
- (Optional) CUDA-compatible GPU

### Installation

```bash
# Clone the repository
git clone https://github.com/hamzasammoud11-dotcom/lacoste001.git
cd lacoste001

# Install Python dependencies
pip install -r bioflow/api/requirements.txt

# Install frontend dependencies
cd lacoste001/ui
pnpm install
cd ../..
```

### Running

**Option 1: Using the launch script (Windows)**

```bash
launch_bioflow_full.bat
```

**Option 2: Manual start**

```bash
# Terminal 1: Start FastAPI backend
python -m uvicorn bioflow.api.server:app --reload --port 8000

# Terminal 2: Start Next.js frontend
cd lacoste001/ui
pnpm dev
```

### Access

- **Frontend**: <http://localhost:3000>
- **API Docs**: <http://localhost:8000/docs>
- **API Health**: <http://localhost:8000/health>

## 📁 Project Structure

```
OpenBioMed/
├── bioflow/                    # BioFlow Platform
│   ├── api/                    # FastAPI Backend
│   │   ├── server.py           # Main API server
│   │   ├── model_service.py    # Unified model access
│   │   ├── qdrant_service.py   # Vector database
│   │   └── dti_predictor.py    # DTI prediction
│   ├── core/                   # Core abstractions
│   ├── plugins/                # Encoders & retrievers
│   └── workflows/              # Pipeline definitions
│
├── lacoste001/
│   └── ui/                     # Next.js Frontend
│       ├── app/
│       │   ├── api/            # API routes
│       │   └── dashboard/      # UI pages
│       ├── components/         # React components
│       └── lib/                # Services & utilities
│
├── open_biomed/                # OpenBioMed Research Engine
│   ├── models/                 # BioT5, ESM, GraphMVP
│   ├── datasets/               # Dataset loaders
│   └── tasks/                  # Task implementations
│
└── configs/                    # YAML configurations
```

## 🔌 API Endpoints

### Discovery Pipeline

- `POST /api/discovery` - Start discovery job
- `GET /api/discovery/{job_id}` - Get job status

### Predictions

- `POST /api/predict` - DTI prediction
- `POST /api/encode` - Encode molecule/protein/text

### Data Management

- `POST /api/ingest` - Add data to vector DB
- `GET /api/molecules` - List molecules
- `GET /api/proteins` - List proteins
- `GET /api/collections` - List vector collections

### Visualization

- `GET /api/explorer/embeddings` - Get 2D projections
- `GET /api/similarity` - Compute similarity scores

## 🧪 Features

### Drug Discovery Pipeline

- Natural language, SMILES, or FASTA input
- Automatic modality detection
- Vector similarity search
- Property prediction (MW, LogP, TPSA)
- Binding affinity prediction

### Molecular Analysis

- 2D/3D molecule visualization
- SMILES validation
- Property calculation via RDKit

### Protein Analysis

- 3D protein structure viewing
- Sequence embedding
- DTI prediction

### Explorer

- UMAP/t-SNE embedding visualization
- Cluster analysis
- Interactive filtering

## 🔧 Configuration

### Environment Variables

```bash
# .env file
NEXT_PUBLIC_API_URL=http://localhost:8000
QDRANT_URL=http://localhost:6333  # Optional: remote Qdrant
QDRANT_PATH=./qdrant_data          # Local Qdrant storage
```

### API Configuration

Edit `lacoste001/ui/config/api.config.ts`:

```typescript
export const API_CONFIG = {
  baseUrl: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  // ...
}
```

## 🧬 Model Support

| Model | Type | Use Case |
|-------|------|----------|
| ChemBERTa | Molecule Encoder | SMILES embeddings |
| ESM-2 | Protein Encoder | Sequence embeddings |
| PubMedBERT | Text Encoder | Biomedical text |
| DeepPurpose | DTI | Binding prediction |
| GraphMVP | Property | Molecular properties |
| BioT5 | Generation | Molecule generation |

## 📊 Development

### Verify Installation

```bash
python scripts/verify_phase3.py
```

### Run Tests

```bash
pytest tests/
```

### Type Checking (Frontend)

```bash
cd lacoste001/ui
pnpm tsc --noEmit
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

- [OpenBioMed](https://github.com/PharMolix/OpenBioMed) - Foundation models
- [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose) - DTI prediction
- [Qdrant](https://qdrant.tech/) - Vector database
- [Next.js](https://nextjs.org/) - React framework
- [Shadcn/ui](https://ui.shadcn.com/) - UI components
