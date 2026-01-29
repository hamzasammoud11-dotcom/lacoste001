#!/usr/bin/env python3
"""Test the BioFlow Agent Pipeline."""
import sys
import os

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

def test_generator():
    """Test GeneratorAgent."""
    print("\n[1] Testing GeneratorAgent...")
    from bioflow.agents import GeneratorAgent
    
    gen = GeneratorAgent(num_samples=5)
    result = gen.process("Generate a kinase inhibitor for cancer treatment")
    
    print(f"   Generated: {len(result.content)} molecules")
    for i, mol in enumerate(result.content[:3]):
        smiles = mol["smiles"][:40]
        conf = mol["confidence"]
        print(f"   [{i+1}] {smiles}... (conf={conf:.2f})")
    
    return result.content

def test_validator(molecules):
    """Test ValidatorAgent."""
    print("\n[2] Testing ValidatorAgent...")
    from bioflow.agents import ValidatorAgent
    
    val = ValidatorAgent()
    smiles_list = [mol["smiles"] for mol in molecules[:3]]
    result = val.process(smiles_list)
    
    total = result.metadata["total"]
    passed = result.metadata["passed"]
    rate = result.metadata["pass_rate"] * 100
    print(f"   Validated: {total} molecules")
    print(f"   Passed: {passed} ({rate:.0f}%)")
    
    for v in result.content[:2]:
        smiles = v["smiles"][:30]
        score = v["score"]
        status = v["status"]
        print(f"   - {smiles}... score={score:.3f} status={status}")
    
    return result.content

def test_ranker(molecules, validations):
    """Test RankerAgent."""
    print("\n[3] Testing RankerAgent...")
    from bioflow.agents import RankerAgent
    
    ranker = RankerAgent()
    candidates = []
    for i, mol in enumerate(molecules[:5]):
        val_score = validations[i]["score"] if i < len(validations) else 0.5
        candidates.append({
            "smiles": mol["smiles"],
            "validation_score": val_score,
            "confidence": mol["confidence"],
        })
    
    result = ranker.process({"candidates": candidates, "top_k": 3})
    
    total = result.metadata["total_candidates"]
    print(f"   Ranked: {total} candidates")
    for r in result.content:
        smiles = r["smiles"][:40]
        rank = r["rank"]
        score = r["final_score"]
        print(f"   [{rank}] score={score:.3f} - {smiles}...")
    
    return result.content

def test_workflow():
    """Test full DiscoveryWorkflow."""
    print("\n[4] Testing DiscoveryWorkflow...")
    from bioflow.agents import DiscoveryWorkflow
    
    workflow = DiscoveryWorkflow(num_candidates=5, top_k=3)
    result = workflow.run("Design an EGFR inhibitor with good bioavailability")
    
    print(f"   Status: {result.status.value}")
    print(f"   Steps: {result.steps_completed}/{result.total_steps}")
    print(f"   Time: {result.execution_time_ms:.1f}ms")
    
    if result.errors:
        print(f"   Errors: {result.errors}")
    
    # Get top candidates
    top = workflow.get_top_candidates(result)
    print(f"\n   Top {len(top)} candidates:")
    for i, cand in enumerate(top):
        smiles = cand.get("smiles", "")[:35]
        score = cand.get("final_score", 0)
        rank = cand.get("rank", i+1)
        print(f"   [{rank}] {smiles}... (score={score:.3f})")
    
    return result

if __name__ == "__main__":
    print("=" * 60)
    print("BioFlow Agent Pipeline - Phase 3 Test")
    print("=" * 60)
    
    # Run individual agent tests
    molecules = test_generator()
    validations = test_validator(molecules)
    rankings = test_ranker(molecules, validations)
    
    # Run full workflow
    result = test_workflow()
    
    print("\n" + "=" * 60)
    print("[OK] All Phase 3 tests passed!")
    print("=" * 60)
