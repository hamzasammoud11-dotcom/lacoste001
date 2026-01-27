#!/usr/bin/env python3
"""Test the BioFlow Agent API endpoints."""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    r = requests.get(f"{BASE_URL}/health")
    print(f"Health: {r.status_code}")
    return r.status_code == 200

def test_generate():
    """Test molecule generation endpoint."""
    print("\n[1] Testing /api/agents/generate...")
    
    data = {
        "prompt": "Design a kinase inhibitor for EGFR",
        "mode": "text",
        "num_samples": 3,
    }
    
    r = requests.post(f"{BASE_URL}/api/agents/generate", json=data)
    print(f"   Status: {r.status_code}")
    
    if r.status_code == 200:
        result = r.json()
        print(f"   ✅ Generated: {len(result.get('molecules', []))} molecules")
        for mol in result.get("molecules", [])[:2]:
            smiles = mol.get("smiles", "")[:40]
            print(f"      - {smiles}...")
        return True
    else:
        print(f"   ❌ Error: {r.text}")
        return False

def test_validate():
    """Test molecule validation endpoint."""
    print("\n[2] Testing /api/agents/validate...")
    
    data = {
        "smiles": [
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # Caffeine
            "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
        ],
    }
    
    r = requests.post(f"{BASE_URL}/api/agents/validate", json=data)
    print(f"   Status: {r.status_code}")
    
    if r.status_code == 200:
        result = r.json()
        summary = result.get("summary", {})
        print(f"   ✅ Validated: {summary.get('total', 0)} molecules")
        print(f"      Passed: {summary.get('passed', 0)}")
        for val in result.get("validations", [])[:2]:
            smiles = val.get("smiles", "")[:25]
            score = val.get("score", 0)
            status = val.get("status", "?")
            print(f"      - {smiles}... score={score:.3f} ({status})")
        return True
    else:
        print(f"   ❌ Error: {r.text}")
        return False

def test_rank():
    """Test candidate ranking endpoint."""
    print("\n[3] Testing /api/agents/rank...")
    
    data = {
        "candidates": [
            {"smiles": "CCO", "validation_score": 0.8, "confidence": 0.9},
            {"smiles": "CC", "validation_score": 0.9, "confidence": 0.7},
            {"smiles": "CCC", "validation_score": 0.7, "confidence": 0.8},
        ],
        "top_k": 2,
    }
    
    r = requests.post(f"{BASE_URL}/api/agents/rank", json=data)
    print(f"   Status: {r.status_code}")
    
    if r.status_code == 200:
        result = r.json()
        ranked = result.get("ranked", [])
        print(f"   ✅ Ranked: {len(ranked)} candidates")
        for cand in ranked:
            print(f"      [{cand.get('rank')}] {cand.get('smiles')} score={cand.get('final_score', 0):.3f}")
        return True
    else:
        print(f"   ❌ Error: {r.text}")
        return False

def test_workflow():
    """Test full discovery workflow endpoint."""
    print("\n[4] Testing /api/agents/workflow...")
    
    data = {
        "query": "Design an EGFR inhibitor with good oral bioavailability",
        "num_candidates": 5,
        "top_k": 3,
    }
    
    r = requests.post(f"{BASE_URL}/api/agents/workflow", json=data)
    print(f"   Status: {r.status_code}")
    
    if r.status_code == 200:
        result = r.json()
        print(f"   ✅ Workflow status: {result.get('status')}")
        print(f"      Steps: {result.get('steps_completed')}/{result.get('total_steps')}")
        print(f"      Time: {result.get('execution_time_ms', 0):.1f}ms")
        
        top = result.get("top_candidates", [])
        print(f"      Top {len(top)} candidates:")
        for cand in top:
            smiles = cand.get("smiles", "")[:30]
            score = cand.get("final_score", 0)
            print(f"         - {smiles}... (score={score:.3f})")
        return True
    else:
        print(f"   ❌ Error: {r.text}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("BioFlow Agent API Test")
    print("=" * 60)
    
    # Wait for server
    import time
    for i in range(10):
        try:
            if test_health():
                break
        except:
            print(f"Waiting for server... ({i+1}/10)")
            time.sleep(1)
    else:
        print("❌ Server not available")
        exit(1)
    
    # Run tests
    test_generate()
    test_validate()
    test_rank()
    test_workflow()
    
    print("\n" + "=" * 60)
    print("✅ Agent API tests complete!")
    print("=" * 60)
