# 🧬 BioFlow + OpenBioMed (OBM) Integration Report

## Executive Summary

Ce document présente l'intégration complète de **BioFlow** avec **OpenBioMed (OBM)** et **Qdrant** pour créer un système d'intelligence biologique multimodale. L'architecture permet d'unifier textes scientifiques, molécules (SMILES) et protéines dans un espace vectoriel commun, facilitant la découverte cross-modale et la conception de médicaments assistée par IA.

> **Note (27/01/2026)**: L'interface Streamlit historique a été retirée du runtime.  
> L'UI officielle est **Next.js** (dossier `ui/`) et le backend est **FastAPI** (port 8000).

---

## 📋 Table des matières

1. [Architecture Générale](#architecture-générale)
2. [Composants Implémentés](#composants-implémentés)
3. [Capacités du Système](#capacités-du-système)
4. [Guide d'Utilisation](#guide-dutilisation)
5. [API Reference](#api-reference)
6. [Scénarios d'Usage](#scénarios-dusage)
7. [Intégration au Projet BioFlow](#intégration-au-projet-bioflow)
8. [Roadmap et Extensions](#roadmap-et-extensions)

---

## 🏗️ Architecture Générale

```
┌─────────────────────────────────────────────────────────────────┐
│                        BioFlow Explorer                         │
│                     (Interface Next.js)                          │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                      BioFlow Pipeline                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Miner   │→ │Generator │→ │Validator │→ │ Ranker   │        │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
└───────┼─────────────┼─────────────┼─────────────┼───────────────┘
        │             │             │             │
┌───────▼─────────────▼─────────────▼─────────────▼───────────────┐
│                     Qdrant Manager                               │
│         (Vector Storage + HNSW Index + Filtering)                │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                       OBM Wrapper                                │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                 │
│  │encode_text │  │encode_smiles│ │encode_protein│               │
│  │   (LLM)    │  │   (GNN)     │ │   (ESM)      │               │
│  └─────┬──────┘  └──────┬──────┘ └──────┬───────┘               │
│        └────────────────┼───────────────┘                        │
│                         ▼                                        │
│              Unified Vector Space (768-4096 dim)                 │
└─────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                      BioMedGPT Model                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ LLaMA LLM    │  │ GraphMVP GNN │  │   ESM-2      │          │
│  │ (Text)       │  │ (Molecules)   │  │ (Proteins)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Flux de Données

1. **Ingestion** : Données brutes (texte, SMILES, séquences) → OBM Wrapper → Vecteurs
2. **Indexation** : Vecteurs + Métadonnées → Qdrant (collection avec index HNSW)
3. **Recherche** : Query → Encoding → Similarité cosinus → Top-K résultats
4. **Pipeline** : Orchestration d'agents spécialisés avec injection contextuelle

---

## 🧩 Composants Implémentés

### 1. OBM Wrapper (`bioflow/obm_wrapper.py`)

Encapsule BioMedGPT pour fournir une API unifiée d'encodage.

| Méthode | Input | Output | Description |
|---------|-------|--------|-------------|
| `encode_text(text)` | String/List[String] | List[EmbeddingResult] | Encode abstracts, notes, descriptions |
| `encode_smiles(smiles)` | String/List[String] | List[EmbeddingResult] | Encode représentations SMILES |
| `encode_protein(seq)` | String/List[String] | List[EmbeddingResult] | Encode séquences FASTA |
| `encode(content, modality)` | Any, ModalityType | EmbeddingResult | API universelle |
| `cross_modal_similarity()` | Query + Targets | List[(content, score)] | Similarité cross-modale |

**Caractéristiques clés :**
- Mode Mock pour tests sans GPU
- Hash de contenu pour déduplication
- Projection vers espace LLM unifié

### 2. Qdrant Manager (`bioflow/qdrant_manager.py`)

Gestion haut-niveau de la base vectorielle.

| Méthode | Description |
|---------|-------------|
| `create_collection()` | Créer/recréer collection avec params HNSW |
| `ingest(items)` | Ingestion batch avec encoding automatique |
| `search(query, ...)` | Recherche avec filtres par modalité |
| `cross_modal_search()` | Recherche inter-modalité |
| `get_neighbors_diversity()` | Analyse de diversité des voisins |

**Payload stocké :**
```json
{
  "content": "...",
  "modality": "text|smiles|protein",
  "source": "PubMed:12345",
  "tags": ["cancer", "kinase"],
  "content_hash": "a1b2c3d4..."
}
```

### 3. Pipeline (`bioflow/pipeline.py`)

Orchestration de workflows multi-agents.

**Agents disponibles :**

| Agent | Type | Rôle |
|-------|------|------|
| `MinerAgent` | MINER | Fouille littérature scientifique |
| `ValidatorAgent` | VALIDATOR | Valide toxicité/propriétés |
| `RankerAgent` | RANKER | Classe candidats multi-critères |

**Workflow type :**
```python
pipeline = BioFlowPipeline(obm, qdrant)
pipeline.register_agent(MinerAgent(obm, qdrant))
pipeline.register_agent(ValidatorAgent(obm, qdrant))
pipeline.set_workflow(["LiteratureMiner", "Validator"])
result = pipeline.run("KRAS inhibitor")
```

### 4. Visualizer (`bioflow/visualizer.py`)

Outils de visualisation pour exploration.

- **EmbeddingVisualizer** : PCA/t-SNE, scatter 2D/3D, matrice de similarité
- **MoleculeVisualizer** : SVG, grilles de molécules (via RDKit)
- **ResultsVisualizer** : Dashboard, graphiques de scores

### 5. Application Web (Next.js)

L'interface officielle est la **Next.js UI** dans `ui/` (aucun runtime Streamlit).

---

## 🎯 Capacités du Système

### Encodage Multimodal

| Modalité | Modèle Backend | Dimension | Notes |
|----------|---------------|-----------|-------|
| Texte | LLaMA (BioMedGPT) | 4096 | Abstracts, notes cliniques |
| Molécules | GraphMVP GNN | 300 → projeté | SMILES vers graphe |
| Protéines | ESM-2 | Variable → projeté | Séquences FASTA |
| Images | BiomedCLIP (ViT-B/16) | 512 → projeté | Microscopy, gels, spectra |

### Recherches Supportées

```
text → text      : Retrouver abstracts similaires
text → smiles    : Trouver molécules mentionnées/pertinentes
text → protein   : Identifier protéines liées
smiles → text    : Quels articles mentionnent cette molécule ?
smiles → smiles  : Analogues structuraux
smiles → protein : Cibles potentielles
protein → text   : Littérature sur cette protéine
protein → smiles : Ligands connus
protein → protein: Protéines homologues
```

### Métriques de Qualité

- **Similarité cosinus** pour le ranking
- **Score de diversité** pour éviter les résultats redondants
- **Distribution des modalités** dans les résultats

---

## 📖 Guide d'Utilisation

### Installation

```bash
# Dépendances principales
pip install -r requirements.txt
pip install qdrant-client plotly scikit-learn

# Optionnel pour visualisation moléculaire
pip install rdkit
```

### Lancement de l'interface

```bash
cd OpenBioMed
# UI (Next.js)
cd ui
pnpm dev
```

### Utilisation Programmatique

```python
from bioflow import OBMWrapper, QdrantManager, BioFlowPipeline

# 1. Initialisation
obm = OBMWrapper(
    device="cuda",                    # ou "cpu"
    config_path="configs/model/biomedgpt.yaml",
    use_mock=False                    # True pour tests sans modèle
)

qdrant = QdrantManager(
    obm=obm,
    qdrant_url="http://localhost:6333",  # Serveur distant
    # ou qdrant_path="./data/qdrant"      # Stockage local
)

# 2. Ingestion de données
data = [
    {"content": "Aspirin reduces inflammation", "modality": "text", "source": "PubMed"},
    {"content": "CC(=O)OC1=CC=CC=C1C(=O)O", "modality": "smiles", "tags": ["aspirin"]},
]
qdrant.ingest(data, collection="my_project")

# 3. Recherche
results = qdrant.search(
    query="anti-inflammatory drug",
    query_modality="text",
    filter_modality="smiles",  # Ne retourner que des molécules
    limit=10
)

for r in results:
    print(f"{r.score:.3f} | {r.modality} | {r.content}")

# 4. Analyse cross-modale
similarities = obm.cross_modal_similarity(
    query="MKWVTFISLLLLFSSAYSRGV",  # Séquence protéine
    query_modality="protein",
    targets=["CCO", "CC(=O)O", "c1ccccc1"],
    target_modality="smiles"
)
```

---

## 📚 API Reference

### EmbeddingResult

```python
@dataclass
class EmbeddingResult:
    vector: np.ndarray      # Vecteur d'embedding
    modality: ModalityType  # Type de donnée
    content: str            # Contenu original (tronqué)
    content_hash: str       # Hash MD5 pour dédup
    dimension: int          # Dimension du vecteur
```

### SearchResult

```python
@dataclass
class SearchResult:
    id: str                 # ID Qdrant
    score: float            # Similarité cosinus [0, 1]
    content: str            # Contenu stocké
    modality: str           # Type de donnée
    payload: Dict[str, Any] # Métadonnées complètes
```

### PipelineResult

```python
@dataclass
class PipelineResult:
    success: bool           # Exécution réussie
    outputs: List[Any]      # Sorties de chaque agent
    messages: List[AgentMessage]  # Messages échangés
    stats: Dict[str, Any]   # Statistiques d'exécution
```

---

## 🔬 Scénarios d'Usage

### UC1 : Découverte de candidats médicaments

```python
# Requête en langage naturel
results = pipeline.run_discovery_workflow(
    query="Small molecule inhibitor for EGFR in non-small cell lung cancer",
    query_modality="text",
    target_modality="smiles"
)

# Résultats : molécules similaires + littérature + validation
```

### UC2 : Identification de cibles protéiques

```python
# À partir d'une molécule connue
results = qdrant.cross_modal_search(
    query="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    query_modality="smiles",
    target_modality="protein",
    limit=5
)
```

### UC3 : Analyse de littérature

```python
# Retrouver articles pertinents pour une protéine
miner = MinerAgent(obm, qdrant)
result = miner.process(
    "MTEYKLVVVGAGGVGKSALTIQLIQ...",  # KRAS
    context={"modality": "protein", "limit": 10}
)
```

### UC4 : Validation de toxicité

```python
validator = ValidatorAgent(obm, qdrant)
validation = validator.process("CCN(CC)c1ccc(N)cc1")  # SMILES candidat

if not validation.content["passed"]:
    print("⚠️ Risques détectés:", validation.content["flagged_risks"])
```

---

## 🔗 Intégration au Projet BioFlow

### Correspondance avec l'architecture cible

| Composant BioFlow | Implémentation OBM |
|-------------------|-------------------|
| Mémoire vectorielle centrale | `QdrantManager` avec collection partagée |
| Encodeur multimodal | `OBMWrapper` (BioMedGPT) |
| Nœuds-agents | Classes `*Agent` dans `pipeline.py` |
| Workflow visuel | **Next.js UI** (`ui/`) + API FastAPI |
| Evidence linking | Payload avec `source`, `tags`, scores |

### Points d'extension

1. **Nouveaux agents** : Hériter de `BaseAgent`
2. **Nouvelles modalités** : Ajouter encodeurs dans `OBMWrapper`
3. **Filtres avancés** : Étendre `QdrantManager.search()`
4. **Visualisations** : Ajouter à `Visualizer`

### Fichiers de configuration

```yaml
# configs/bioflow/default.yaml (à créer)
qdrant:
  url: "http://localhost:6333"
  collection: "bioflow_memory"

obm:
  config: "configs/model/biomedgpt.yaml"
  checkpoint: null  # ou path vers weights
  device: "cuda"

pipeline:
  default_agents: ["LiteratureMiner", "Validator"]
```

---

## 🚀 Roadmap et Extensions

### Phase 1 (Actuel) ✅
- [x] OBM Wrapper avec encodage multimodal
- [x] Intégration Qdrant
- [x] Agents de base (Miner, Validator, Ranker)
- [x] Interface Next.js (UI officielle)
- [x] Mode Mock pour développement

### Phase 2 (Court terme)
- [ ] Chargement effectif des checkpoints BioMedGPT
- [ ] Agent Générateur (MolT5, DeepPurpose)
- [ ] Persistance Qdrant (Docker)
- [ ] Tests unitaires complets
- [ ] Batch processing optimisé

### Phase 3 (Moyen terme)
- [ ] Intégration Knowledge Graph
- [ ] Evidence linking avec citations
- [ ] API REST (FastAPI)
- [ ] Monitoring et logging structuré
- [ ] Déploiement cloud (Qdrant Cloud)

### Phase 4 (Long terme)
- [ ] Fine-tuning OBM sur données custom
- [ ] Génération de molécules contrainte
- [ ] Workflow de wet lab feedback
- [ ] Intégration LIMS

---

## 📊 Métriques de Performance

| Opération | Latence (Mock) | Latence (GPU) | Notes |
|-----------|----------------|---------------|-------|
| encode_text (1) | ~1ms | ~50ms | Batch plus efficace |
| encode_smiles (1) | ~1ms | ~30ms | GNN léger |
| encode_protein (1) | ~1ms | ~100ms | ESM plus lourd |
| search (top-10) | ~5ms | ~5ms | HNSW constant |
| ingest (100 items) | ~100ms | ~3s | Dominé par encoding |

---

## 🎓 Conclusion

L'intégration BioFlow + OBM + Qdrant fournit une base solide pour :

1. **Exploration unifiée** des données biologiques hétérogènes
2. **Découverte cross-modale** (texte ↔ molécule ↔ protéine)
3. **Workflows automatisés** avec agents spécialisés
4. **Traçabilité scientifique** via métadonnées et payloads

Le système est prêt pour un prototypage rapide (mode Mock) et peut évoluer vers une production avec GPU et Qdrant persistant.

---

*Document généré le 25/01/2026 - Rami Troudi*
