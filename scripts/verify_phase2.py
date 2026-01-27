"""
Phase 2 Verification Script
=============================

Tests the real open-source encoder implementations.

Usage:
    python scripts/verify_phase2.py [--full]
    
    --full: Run full tests with model downloads (slow, requires GPU recommended)
    
Without --full, runs quick tests with fallback/mock behavior.
"""

import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_registry():
    """Test that all plugins are importable and registry works."""
    print("\n" + "="*60)
    print("🛠️  TEST 1: Plugin Registry")
    print("="*60)
    
    from bioflow.core import ToolRegistry, Modality
    from bioflow.plugins import (
        OBMEncoder,
        TextEncoder,
        MoleculeEncoder,
        ProteinEncoder,
        QdrantRetriever,
        DeepPurposePredictor
    )
    
    print("✅ All plugins imported successfully")
    
    # List available
    print("\nAvailable plugins:")
    print("  • OBMEncoder (multimodal)")
    print("  • TextEncoder (PubMedBERT/SciBERT)")
    print("  • MoleculeEncoder (ChemBERTa/RDKit)")
    print("  • ProteinEncoder (ESM-2/ProtBERT)")
    print("  • QdrantRetriever (Qdrant)")
    print("  • DeepPurposePredictor (DTI)")
    
    return True


def test_rdkit_fallback():
    """Test RDKit molecule encoding (no GPU needed)."""
    print("\n" + "="*60)
    print("🧪 TEST 2: RDKit Molecule Encoder (CPU-only)")
    print("="*60)
    
    try:
        from bioflow.plugins.encoders.molecule_encoder import MoleculeEncoder
        from bioflow.core import Modality
        
        encoder = MoleculeEncoder(backend="rdkit_morgan", fp_size=2048)
        
        test_molecules = [
            ("CCO", "Ethanol"),
            ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
        ]
        
        print(f"Encoder dimension: {encoder.dimension}")
        print("\nEncoding molecules:")
        
        for smiles, name in test_molecules:
            result = encoder.encode(smiles, Modality.SMILES)
            nonzero = sum(1 for v in result.vector if v > 0)
            print(f"  • {name}: {nonzero} bits set (of {len(result.vector)})")
        
        print("✅ RDKit encoding works!")
        return True
        
    except ImportError as e:
        print(f"⚠️  RDKit not installed: {e}")
        print("   Install with: pip install rdkit")
        return False


def test_deeppurpose_predictor():
    """Test DeepPurpose predictor (with fallback)."""
    print("\n" + "="*60)
    print("🔮 TEST 3: DeepPurpose Predictor")
    print("="*60)
    
    from bioflow.plugins.deeppurpose_predictor import DeepPurposePredictor
    
    predictor = DeepPurposePredictor()
    
    # Test data
    drug = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    target = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    print(f"Drug: Aspirin (SMILES: {drug[:30]}...)")
    print(f"Target: Protein ({len(target)} amino acids)")
    
    result = predictor.predict(drug, target)
    
    print(f"\nPrediction:")
    print(f"  • Score: {result.score:.3f}")
    print(f"  • Label: {result.label}")
    print(f"  • Confidence: {result.confidence:.2f}")
    print(f"  • Method: {result.metadata.get('method', 'unknown')}")
    
    if result.metadata.get('warning'):
        print(f"  ⚠️  {result.metadata['warning']}")
    
    print("✅ Predictor works (with fallback if DeepPurpose unavailable)")
    return True


def test_qdrant_retriever():
    """Test Qdrant retriever with mock encoder."""
    print("\n" + "="*60)
    print("🗄️  TEST 4: Qdrant Retriever (In-Memory)")
    print("="*60)
    
    try:
        from bioflow.core import BioEncoder, Modality, EmbeddingResult
        from bioflow.plugins.qdrant_retriever import QdrantRetriever
        
        # Mock encoder for testing
        class MockEncoder(BioEncoder):
            def encode(self, content, modality):
                # Simple hash-based vector
                import hashlib
                h = hashlib.md5(content.encode()).hexdigest()
                vector = [int(c, 16) / 15.0 for c in h] * 48  # 768-dim
                return EmbeddingResult(vector=vector[:768], modality=modality, dimension=768)
            
            @property
            def dimension(self): return 768
        
        encoder = MockEncoder()
        retriever = QdrantRetriever(encoder=encoder, collection="test_molecules")
        
        # Ingest test data
        test_data = [
            ("CCO", "Ethanol", {"type": "alcohol"}),
            ("CCCO", "Propanol", {"type": "alcohol"}),
            ("CC(=O)O", "Acetic acid", {"type": "acid"}),
            ("c1ccccc1", "Benzene", {"type": "aromatic"}),
        ]
        
        print("Ingesting molecules...")
        for smiles, name, payload in test_data:
            retriever.ingest(smiles, Modality.SMILES, {"name": name, **payload})
        
        print(f"Collection size: {retriever.count()}")
        
        # Search
        print("\nSearching for 'CCCCO' (Butanol)...")
        results = retriever.search("CCCCO", limit=3, modality=Modality.SMILES)
        
        print("Results:")
        for r in results:
            print(f"  • {r.payload.get('name', 'Unknown')}: score={r.score:.3f}")
        
        print("✅ Qdrant retriever works!")
        return True
        
    except ImportError as e:
        print(f"⚠️  qdrant-client not installed: {e}")
        print("   Install with: pip install qdrant-client")
        return False


def test_full_obm_encoder():
    """Test full OBM encoder with real models (slow, requires downloads)."""
    print("\n" + "="*60)
    print("🚀 TEST 5: Full OBM Encoder (requires model downloads)")
    print("="*60)
    
    try:
        from bioflow.plugins.obm_encoder import OBMEncoder
        from bioflow.core import Modality
        
        print("Initializing OBMEncoder...")
        print("(This will download models on first run - ~500MB)")
        
        obm = OBMEncoder(
            text_model="pubmedbert",
            molecule_model="chemberta",
            protein_model="esm2_t6",  # Smallest ESM model
            lazy_load=True
        )
        
        # Test text
        print("\n1. Encoding text...")
        text_result = obm.encode("EGFR inhibitor for lung cancer treatment", Modality.TEXT)
        print(f"   Text embedding: {len(text_result.vector)} dims")
        
        # Test molecule
        print("2. Encoding molecule...")
        mol_result = obm.encode("CC(=O)Oc1ccccc1C(=O)O", Modality.SMILES)
        print(f"   Molecule embedding: {len(mol_result.vector)} dims")
        
        # Test protein
        print("3. Encoding protein...")
        prot_result = obm.encode("MKTVRQERLKSIVRILERSKEPVSG", Modality.PROTEIN)
        print(f"   Protein embedding: {len(prot_result.vector)} dims")
        
        # Cross-modal similarity
        print("\n4. Cross-modal similarity:")
        sim = obm.similarity(text_result, mol_result)
        print(f"   Text-Molecule similarity: {sim:.3f}")
        
        print("\n✅ Full OBM Encoder works!")
        print(obm.get_encoder_info())
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run verification tests."""
    print("="*60)
    print("🧬 BioFlow Phase 2 Verification")
    print("="*60)
    
    full_mode = "--full" in sys.argv
    
    if full_mode:
        print("Running FULL tests (with model downloads)")
    else:
        print("Running QUICK tests (no model downloads)")
        print("Add --full flag for complete testing")
    
    results = {}
    
    # Always run
    results["Registry"] = test_registry()
    results["RDKit"] = test_rdkit_fallback()
    results["DeepPurpose"] = test_deeppurpose_predictor()
    results["Qdrant"] = test_qdrant_retriever()
    
    # Only in full mode
    if full_mode:
        results["OBMEncoder"] = test_full_obm_encoder()
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("✅ All tests passed!" if all_passed else "⚠️  Some tests failed"))
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
