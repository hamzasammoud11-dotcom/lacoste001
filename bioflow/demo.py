"""
BioFlow Demo Script - Test all capabilities
=============================================

This script demonstrates all major features of the BioFlow system.
Run this to verify your installation and see the system in action.

Usage:
    python bioflow/demo.py
"""

import os
import sys
import numpy as np
from pprint import pprint

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def demo_obm_encoding():
    """Demonstrate OBM encoding capabilities."""
    print_header("🧬 OBM Multimodal Encoding")
    
    from bioflow.obm_wrapper import OBMWrapper, ModalityType
    
    # Initialize OBMWrapper (model must be properly configured)
    obm = OBMWrapper()
    print(f"✅ OBM initialized")
    print(f"   Vector dimension: {obm.vector_dim}")
    print(f"   Device: {obm.device}")
    
    # Encode text
    print("\n📝 Encoding Text:")
    texts = [
        "KRAS is a protein involved in cell signaling",
        "Aspirin is used to reduce inflammation"
    ]
    text_embeddings = obm.encode_text(texts)
    for emb in text_embeddings:
        print(f"   [{emb.modality.value}] dim={emb.dimension}, hash={emb.content_hash}")
        print(f"   Content: {emb.content[:50]}...")
    
    # Encode SMILES
    print("\n🧪 Encoding SMILES:")
    smiles_list = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CCO",                         # Ethanol
        "c1ccccc1"                     # Benzene
    ]
    mol_embeddings = obm.encode_smiles(smiles_list)
    for emb in mol_embeddings:
        print(f"   [{emb.modality.value}] {emb.content} → dim={emb.dimension}")
    
    # Encode proteins
    print("\n🔬 Encoding Proteins:")
    proteins = [
        "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGET",  # KRAS fragment
    ]
    prot_embeddings = obm.encode_protein(proteins)
    for emb in prot_embeddings:
        print(f"   [{emb.modality.value}] {emb.content[:30]}... → dim={emb.dimension}")
    
    # Cross-modal similarity
    print("\n🔄 Cross-Modal Similarity (Text → Molecules):")
    similarities = obm.cross_modal_similarity(
        query="anti-inflammatory drug",
        query_modality="text",
        targets=smiles_list,
        target_modality="smiles"
    )
    for content, score in similarities:
        print(f"   {score:.4f} | {content}")
    
    return obm


def demo_qdrant_manager(obm):
    """Demonstrate Qdrant vector storage."""
    print_header("📦 Qdrant Vector Storage")
    
    from bioflow.qdrant_manager import QdrantManager
    
    # Initialize with in-memory storage
    qdrant = QdrantManager(obm, default_collection="demo_collection")
    print(f"✅ Qdrant Manager initialized (in-memory)")
    
    # Create collection
    qdrant.create_collection(recreate=True)
    print(f"   Collection created: demo_collection")
    
    # Ingest sample data
    print("\n📥 Ingesting Sample Data:")
    sample_data = [
        {"content": "Aspirin is used to reduce fever and relieve mild to moderate pain", "modality": "text", "source": "PubMed:001", "tags": ["pain", "fever"]},
        {"content": "CC(=O)OC1=CC=CC=C1C(=O)O", "modality": "smiles", "source": "ChEMBL", "tags": ["aspirin", "nsaid"]},
        {"content": "Ibuprofen is a nonsteroidal anti-inflammatory drug", "modality": "text", "source": "PubMed:002", "tags": ["nsaid"]},
        {"content": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "modality": "smiles", "source": "ChEMBL", "tags": ["ibuprofen"]},
        {"content": "KRAS mutations are found in many cancers", "modality": "text", "source": "PubMed:003", "tags": ["cancer", "KRAS"]},
        {"content": "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGET", "modality": "protein", "source": "UniProt:P01116", "tags": ["KRAS"]},
    ]
    
    stats = qdrant.ingest(sample_data)
    print(f"   Ingestion stats: {stats}")
    
    # Collection info
    info = qdrant.get_collection_info()
    print(f"\n📊 Collection Info:")
    for k, v in info.items():
        print(f"   {k}: {v}")
    
    # Search
    print("\n🔍 Searching for 'anti-inflammatory':")
    results = qdrant.search(
        query="anti-inflammatory medicine",
        query_modality="text",
        limit=3
    )
    for r in results:
        print(f"   {r.score:.4f} | [{r.modality}] {r.content[:50]}...")
    
    # Cross-modal search
    print("\n🔄 Cross-Modal Search (Text → Molecules):")
    results = qdrant.cross_modal_search(
        query="pain relief medication",
        query_modality="text",
        target_modality="smiles",
        limit=3
    )
    for r in results:
        print(f"   {r.score:.4f} | {r.content}")
    
    # Diversity analysis
    print("\n📈 Neighbors Diversity Analysis:")
    diversity = qdrant.get_neighbors_diversity(
        query="cancer treatment",
        query_modality="text",
        k=5
    )
    for k, v in diversity.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")
    
    return qdrant


def demo_pipeline(obm, qdrant):
    """Demonstrate the pipeline system."""
    print_header("🔬 BioFlow Pipeline")
    
    from bioflow.pipeline import BioFlowPipeline, MinerAgent, ValidatorAgent
    
    # Initialize pipeline
    pipeline = BioFlowPipeline(obm, qdrant)
    print("✅ Pipeline initialized")
    
    # Register agents
    miner = MinerAgent(obm, qdrant, "demo_collection")
    validator = ValidatorAgent(obm, qdrant, "demo_collection")
    
    pipeline.register_agent(miner)
    pipeline.register_agent(validator)
    print(f"   Registered agents: {list(pipeline.agents.keys())}")
    
    # Run discovery workflow
    print("\n🚀 Running Discovery Workflow:")
    print("   Query: 'anti-inflammatory drug for pain'")
    
    results = pipeline.run_discovery_workflow(
        query="anti-inflammatory drug for pain",
        query_modality="text",
        target_modality="smiles"
    )
    
    print("\n   📚 Literature Results:")
    for item in results["stages"].get("literature", [])[:3]:
        print(f"      {item['score']:.4f} | {item['content'][:40]}...")
    
    print("\n   🧪 Molecule Results:")
    for item in results["stages"].get("molecules", [])[:3]:
        print(f"      {item['score']:.4f} | {item['content']}")
    
    print("\n   📊 Diversity:")
    div = results["stages"].get("diversity", {})
    print(f"      Mean similarity: {div.get('mean_similarity', 0):.4f}")
    print(f"      Diversity score: {div.get('diversity_score', 0):.4f}")


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print_header("📊 Visualization Capabilities")
    
    try:
        from bioflow.visualizer import EmbeddingVisualizer, ResultsVisualizer
        print("✅ Visualization module loaded")
        
        # Generate sample embeddings
        n_samples = 20
        embeddings = np.random.randn(n_samples, 768)
        labels = [f"Sample {i}" for i in range(n_samples)]
        colors = ["text"] * 7 + ["smiles"] * 7 + ["protein"] * 6
        
        # Dimensionality reduction
        print("\n🔻 Dimensionality Reduction:")
        reduced = EmbeddingVisualizer.reduce_dimensions(embeddings, method="pca", n_components=2)
        print(f"   Original shape: {embeddings.shape}")
        print(f"   Reduced shape: {reduced.shape}")
        
        # Note about plots
        print("\n📈 Plotting Functions Available:")
        print("   - plot_embeddings_2d(embeddings, labels, colors)")
        print("   - plot_embeddings_3d(embeddings, labels)")
        print("   - plot_similarity_matrix(embeddings, labels)")
        print("   - create_dashboard(results, embeddings)")
        print("\n   Use the Next.js UI for interactive visualizations:")
        print("     - Start backend: python -m uvicorn bioflow.api.server:app --host 0.0.0.0 --port 8000")
        print("     - Start UI:      cd ui && pnpm dev")
        
    except ImportError as e:
        print(f"⚠️ Some visualization dependencies missing: {e}")
        print("   Install with: pip install plotly scikit-learn")


def main():
    """Run all demos."""
    print("\n" + "🧬" * 20)
    print("   BIOFLOW + OBM DEMO")
    print("🧬" * 20)
    
    print("\nThis demo requires the OBM model to be properly configured.")
    print("Ensure checkpoints are downloaded and GPU is available.\n")
    
    # Run demos
    obm = demo_obm_encoding()
    qdrant = demo_qdrant_manager(obm)
    demo_pipeline(obm, qdrant)
    demo_visualization()
    
    print_header("✅ Demo Complete!")
    print("Next steps:")
    print("  1. Run the full stack:")
    print("     - Backend: python -m uvicorn bioflow.api.server:app --host 0.0.0.0 --port 8000")
    print("     - UI:      cd ui && pnpm dev")
    print("")
    print("  2. Ensure OBM model is configured:")
    print("     - BioMedGPT checkpoints are downloaded")
    print("     - GPU is available")
    print("")
    print("  3. Read the documentation:")
    print("     docs/BIOFLOW_OBM_REPORT.md")
    print("")


if __name__ == "__main__":
    main()
