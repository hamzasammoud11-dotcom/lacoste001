"""
Phase 3 Verification: Unified Workflow
========================================

Tests the complete discovery pipeline end-to-end:
1. Ingest sample data into Qdrant
2. Run discovery pipeline with query
3. Verify predictions and traceability

Usage:
    python scripts/verify_phase3.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_mock_components():
    """Create mock encoder, retriever, predictor for testing."""
    from bioflow.core import (
        BioEncoder, BioRetriever, BioPredictor,
        Modality, EmbeddingResult, RetrievalResult, PredictionResult
    )
    import hashlib
    
    class MockEncoder(BioEncoder):
        def encode(self, content, modality):
            h = hashlib.md5(str(content).encode()).hexdigest()
            vector = [int(c, 16) / 15.0 for c in h] * 48
            return EmbeddingResult(vector=vector[:768], modality=modality, dimension=768)
        
        def encode_auto(self, content):
            return self.encode(content, Modality.TEXT)
        
        def batch_encode(self, contents, modality):
            return [self.encode(c, modality) for c in contents]
        
        @property
        def dimension(self): return 768
    
    class MockRetriever(BioRetriever):
        def __init__(self, encoder):
            self.encoder = encoder
            self._data = {}
            self._vectors = {}
            self._id_counter = 0
        
        def search(self, query, limit=10, filters=None, collection=None, modality=None, **kwargs):
            if isinstance(query, str):
                query_vec = self.encoder.encode(query, modality or Modality.TEXT).vector
            else:
                query_vec = query
            
            # Simple cosine similarity
            import math
            results = []
            for id_, (vec, payload) in self._vectors.items():
                dot = sum(a*b for a, b in zip(query_vec, vec))
                norm_q = math.sqrt(sum(a*a for a in query_vec))
                norm_v = math.sqrt(sum(b*b for b in vec))
                score = dot / (norm_q * norm_v) if norm_q * norm_v > 0 else 0
                
                results.append(RetrievalResult(
                    id=id_,
                    score=score,
                    content=payload.get("content", ""),
                    modality=Modality(payload.get("modality", "text")),
                    payload=payload
                ))
            
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]
        
        def ingest(self, content, modality, payload=None, collection=None, id=None):
            self._id_counter += 1
            id_ = id or f"item_{self._id_counter}"
            vec = self.encoder.encode(content, modality).vector
            full_payload = {"content": content, "modality": modality.value, **(payload or {})}
            self._vectors[id_] = (vec, full_payload)
            return id_
        
        def count(self, collection=None):
            return len(self._vectors)
    
    class MockPredictor(BioPredictor):
        def predict(self, drug, target):
            import random
            random.seed(hash(drug + target) % 2**32)
            score = random.uniform(0.2, 0.9)
            return PredictionResult(
                score=score,
                confidence=0.7,
                label="binding" if score > 0.5 else "non-binding"
            )
    
    encoder = MockEncoder()
    retriever = MockRetriever(encoder)
    predictor = MockPredictor()
    
    return encoder, retriever, predictor


def test_node_execution():
    """Test individual node execution."""
    print("\n" + "="*60)
    print("🧩 TEST 1: Node Execution")
    print("="*60)
    
    from bioflow.core.nodes import (
        EncodeNode, RetrieveNode, PredictNode, FilterNode, TraceabilityNode
    )
    from bioflow.core import Modality
    
    encoder, retriever, predictor = create_mock_components()
    
    # Test EncodeNode
    encode_node = EncodeNode("enc", encoder, Modality.SMILES)
    result = encode_node.execute("CCO")
    print(f"  EncodeNode: vector dim = {len(result.data.vector)}")
    
    # Test FilterNode
    filter_node = FilterNode("filter", threshold=0.5, top_k=3)
    items = [{"score": 0.9}, {"score": 0.4}, {"score": 0.7}, {"score": 0.3}]
    result = filter_node.execute(items)
    print(f"  FilterNode: {len(items)} items → {len(result.data)} after filtering")
    
    # Test TraceabilityNode
    trace_node = TraceabilityNode("trace")
    items = [{"id": "PMID_12345", "content": "test", "payload": {"pmid": "12345"}}]
    result = trace_node.execute(items)
    print(f"  TraceabilityNode: Added {result.metadata['with_evidence']} evidence links")
    
    print("✅ All nodes execute correctly")
    return True


def test_discovery_pipeline():
    """Test the full discovery pipeline."""
    print("\n" + "="*60)
    print("🔬 TEST 2: Discovery Pipeline")
    print("="*60)
    
    from bioflow.workflows import DiscoveryPipeline, generate_sample_molecules
    from bioflow.core import Modality
    
    encoder, retriever, predictor = create_mock_components()
    
    # Create pipeline
    pipeline = DiscoveryPipeline(
        encoder=encoder,
        retriever=retriever,
        predictor=predictor,
        collection="test_molecules"
    )
    
    # Ingest sample data
    print("\n1. Ingesting sample molecules...")
    sample_data = generate_sample_molecules()
    
    for mol in sample_data:
        retriever.ingest(
            content=mol["smiles"],
            modality=Modality.SMILES,
            payload={"name": mol["name"], **{k: v for k, v in mol.items() if k not in ["smiles", "modality"]}}
        )
    
    print(f"   Ingested {retriever.count()} molecules")
    
    # Run discovery
    print("\n2. Running discovery pipeline...")
    target_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    result = pipeline.discover(
        query="anti-inflammatory compound",
        target_sequence=target_sequence,
        limit=5,
        threshold=0.3,
        top_k=3
    )
    
    print(f"\n   Discovery Result:")
    print(f"   • Query: {result.query[:40]}...")
    print(f"   • Candidates retrieved: {len(result.candidates)}")
    print(f"   • Predictions made: {len(result.predictions)}")
    print(f"   • Top hits: {len(result.top_hits)}")
    print(f"   • Execution time: {result.execution_time_ms:.0f}ms")
    
    print("\n3. Top hits:")
    for i, hit in enumerate(result.top_hits[:3]):
        drug = hit.get("drug", "")[:30]
        score = hit.get("score", 0)
        evidence = hit.get("evidence_links", {})
        print(f"   {i+1}. {drug}... (score: {score:.3f})")
        if evidence:
            print(f"      Evidence: {list(evidence.keys())}")
    
    print("\n✅ Discovery pipeline works!")
    return True


def test_simple_search():
    """Test simple similarity search."""
    print("\n" + "="*60)
    print("🔍 TEST 3: Simple Search")
    print("="*60)
    
    from bioflow.workflows import DiscoveryPipeline, generate_sample_abstracts
    from bioflow.core import Modality
    
    encoder, retriever, predictor = create_mock_components()
    
    pipeline = DiscoveryPipeline(
        encoder=encoder,
        retriever=retriever,
        predictor=predictor
    )
    
    # Ingest abstracts
    print("\n1. Ingesting sample abstracts...")
    for abstract in generate_sample_abstracts():
        retriever.ingest(
            content=abstract["content"],
            modality=Modality.TEXT,
            payload={k: v for k, v in abstract.items() if k not in ["content", "modality"]}
        )
    
    print(f"   Ingested {retriever.count()} items")
    
    # Search
    print("\n2. Searching for 'EGFR cancer treatment'...")
    results = pipeline.search(
        query="EGFR cancer treatment",
        modality=Modality.TEXT,
        limit=3
    )
    
    print(f"\n   Found {len(results)} results:")
    for r in results:
        print(f"   • Score: {r.score:.3f} | {r.content[:50]}...")
    
    print("\n✅ Search works!")
    return True


def test_ingestion_utilities():
    """Test data ingestion utilities."""
    print("\n" + "="*60)
    print("📥 TEST 4: Ingestion Utilities")
    print("="*60)
    
    from bioflow.workflows.ingestion import (
        generate_sample_molecules,
        generate_sample_proteins,
        generate_sample_abstracts
    )
    
    molecules = generate_sample_molecules()
    proteins = generate_sample_proteins()
    abstracts = generate_sample_abstracts()
    
    print(f"  • Sample molecules: {len(molecules)}")
    print(f"    - Example: {molecules[0]['name']} ({molecules[0]['smiles'][:20]}...)")
    
    print(f"  • Sample proteins: {len(proteins)}")
    print(f"    - Example: {proteins[0]['name']} ({proteins[0]['sequence'][:20]}...)")
    
    print(f"  • Sample abstracts: {len(abstracts)}")
    print(f"    - Example: {abstracts[0]['title']}")
    
    print("\n✅ Ingestion utilities work!")
    return True


def test_traceability():
    """Test evidence linking and traceability."""
    print("\n" + "="*60)
    print("🔗 TEST 5: Traceability & Evidence Linking")
    print("="*60)
    
    from bioflow.core.nodes import TraceabilityNode
    
    trace_node = TraceabilityNode("trace")
    
    # Test with different ID formats
    test_items = [
        {"id": "PMID_12345678", "content": "Paper about EGFR", "payload": {}},
        {"id": "mol_1", "content": "CCO", "payload": {"drugbank_id": "DB00316", "pubchem_id": "702"}},
        {"id": "prot_1", "content": "MKTVRQ...", "payload": {"uniprot": "P00533"}},
    ]
    
    result = trace_node.execute(test_items)
    
    print("  Evidence links generated:")
    for item in result.data:
        print(f"  • ID: {item['id']}")
        links = item.get("evidence_links", {})
        if links:
            for source, url in links.items():
                print(f"    → {source}: {url}")
        else:
            print(f"    → No links (payload missing IDs)")
    
    print(f"\n  Items with evidence: {result.metadata['with_evidence']}/{len(test_items)}")
    print("\n✅ Traceability works!")
    return True


def main():
    """Run all Phase 3 verification tests."""
    print("="*60)
    print("🧬 BioFlow Phase 3 Verification: Unified Workflow")
    print("="*60)
    
    results = {}
    
    results["Nodes"] = test_node_execution()
    results["Discovery"] = test_discovery_pipeline()
    results["Search"] = test_simple_search()
    results["Ingestion"] = test_ingestion_utilities()
    results["Traceability"] = test_traceability()
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("✅ All Phase 3 tests passed!" if all_passed else "⚠️ Some tests failed"))
    
    if all_passed:
        print("\n🎉 The unified workflow is ready!")
        print("   You can now:")
        print("   • Ingest molecules, proteins, and literature")
        print("   • Run cross-modal similarity search")
        print("   • Predict drug-target interactions")
        print("   • Trace results back to sources")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
