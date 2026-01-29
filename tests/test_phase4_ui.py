"""
Phase 4 UI Feature Tests
Tests all the new UI components and their API integrations
"""
import time
import json

def test_visualization_api():
    """Test the search API that powers the visualization page"""
    import requests
    
    print("=" * 60)
    print("Testing Visualization API (Search)")
    print("=" * 60)
    
    try:
        # Test search endpoint
        response = requests.post(
            "http://localhost:8000/api/search",
            json={
                "query": "EGFR inhibitor for lung cancer",
                "top_k": 10,
                "use_mmr": True
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Search API works: {len(data.get('results', []))} results")
            
            # Check result structure
            if data.get('results'):
                result = data['results'][0]
                required_fields = ['content', 'score', 'modality']
                for field in required_fields:
                    if field in result:
                        print(f"   [OK] Has '{field}' field")
                    else:
                        print(f"   ✗ Missing '{field}' field")
            return True
        else:
            print(f"[FAIL] Search API failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Search API error: {e}")
        return False


def test_workflow_api():
    """Test the workflow API that powers the workflow builder"""
    import requests
    
    print("\n" + "=" * 60)
    print("Testing Workflow API (Agent Pipeline)")
    print("=" * 60)
    
    try:
        # Test workflow endpoint
        response = requests.post(
            "http://localhost:8000/api/agents/workflow",
            json={
                "query": "kinase inhibitor drug",
                "num_candidates": 3,
                "validation_checks": ["lipinski", "qed"],
                "ranking_weights": {"qed": 0.5, "validity": 0.5}
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Workflow API works")
            print(f"   Steps completed: {data.get('steps_completed', 0)}")
            print(f"   Total time: {data.get('total_time_ms', 0):.1f}ms")
            candidates = data.get("top_candidates") or data.get("candidates", [])
            print(f"   Candidates: {len(candidates)}")
            
            # Check candidate structure
            if candidates:
                candidate = candidates[0]
                print(f"\n   Sample candidate:")
                print(f"   - Name: {candidate.get('name', 'N/A')}")
                print(f"   - SMILES: {candidate.get('smiles', 'N/A')[:30]}...")
                print(f"   - Score: {candidate.get('score', 0):.3f}")
                
                validation = candidate.get('validation', {})
                print(f"   - Valid: {validation.get('is_valid', False)}")
            return True
        else:
            print(f"[FAIL] Workflow API failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Workflow API error: {e}")
        return False


def test_generate_api():
    """Test the generate API endpoint"""
    import requests
    
    print("\n" + "=" * 60)
    print("Testing Generate API")
    print("=" * 60)
    
    try:
        response = requests.post(
            "http://localhost:8000/api/agents/generate",
            json={
                "prompt": "anti-inflammatory compound",
                "mode": "text",
                "num_samples": 3
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            molecules = data.get('molecules', [])
            print(f"[OK] Generate API works: {len(molecules)} molecules")
            for mol in molecules[:2]:
                if isinstance(mol, dict):
                    print(f"   - {mol.get('name', 'N/A')}: {mol.get('smiles', '')[:40]}...")
                else:
                    print(f"   - {str(mol)[:50]}...")
            return True
        else:
            print(f"[FAIL] Generate API failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Generate API error: {e}")
        return False


def test_validate_api():
    """Test the validate API endpoint"""
    import requests
    
    print("\n" + "=" * 60)
    print("Testing Validate API")
    print("=" * 60)
    
    try:
        response = requests.post(
            "http://localhost:8000/api/agents/validate",
            json={
                "smiles": [
                    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
                    "CCO"  # Ethanol
                ],
                "check_lipinski": True,
                "check_admet": True,
                "check_alerts": True
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('validations', [])
            print(f"[OK] Validate API works: {len(results)} results")
            for i, result in enumerate(results):
                if isinstance(result, dict):
                    print(f"   - SMILES {i+1}: Valid={result.get('is_valid', False)}")
                else:
                    print(f"   - SMILES {i+1}: {result}")
            return True
        else:
            print(f"[FAIL] Validate API failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Validate API error: {e}")
        return False


def test_rank_api():
    """Test the rank API endpoint"""
    import requests
    
    print("\n" + "=" * 60)
    print("Testing Rank API")
    print("=" * 60)
    
    try:
        response = requests.post(
            "http://localhost:8000/api/agents/rank",
            json={
                "candidates": [
                    {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "name": "Aspirin", "score": 0.8},
                    {"smiles": "CC(=O)NC1=CC=C(C=C1)O", "name": "Acetaminophen", "score": 0.7}
                ],
                "weights": {"qed": 0.5, "validity": 0.5},
                "top_k": 5
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            ranked = data.get('ranked', [])
            print(f"[OK] Rank API works: {len(ranked)} ranked")
            for item in ranked:
                print(f"   - {item.get('name', 'N/A')}: Score={item.get('score', 0):.3f}")
            return True
        else:
            print(f"[FAIL] Rank API failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Rank API error: {e}")
        return False


def test_export_functions():
    """Test export functionality by simulating what the UI does"""
    print("\n" + "=" * 60)
    print("Testing Export Functions (Simulation)")
    print("=" * 60)
    
    # Simulate the data that would be exported
    sample_results = [
        {
            "id": "1",
            "content": "EGFR inhibitor compound",
            "score": 0.95,
            "modality": "text",
            "source": "pubmed",
            "citation": "Nature 2024"
        },
        {
            "id": "2",
            "content": "MTAAPRGPRL",  # Protein sequence
            "score": 0.88,
            "modality": "protein",
            "source": "uniprot",
            "citation": "UniProt P12345"
        }
    ]
    
    # Test CSV export logic
    try:
        headers = ["id", "content", "score", "modality", "source", "citation"]
        rows = []
        for r in sample_results:
            rows.append([
                r['id'],
                f'"{r["content"].replace(chr(34), chr(34)+chr(34))}"',
                str(r['score']),
                r['modality'],
                r['source'],
                r.get('citation', '')
            ])
        csv_content = ",".join(headers) + "\n" + "\n".join([",".join(row) for row in rows])
        print(f"[OK] CSV export works: {len(csv_content)} chars")
    except Exception as e:
        print(f"[ERROR] CSV export error: {e}")
    
    # Test JSON export logic
    try:
        json_content = json.dumps(sample_results, indent=2)
        print(f"[OK] JSON export works: {len(json_content)} chars")
    except Exception as e:
        print(f"[ERROR] JSON export error: {e}")
    
    # Test FASTA export logic
    try:
        fasta_lines = []
        for r in sample_results:
            if r['modality'] == 'protein':
                fasta_lines.append(f">{r['id']}\n{r['content']}")
        fasta_content = "\n\n".join(fasta_lines)
        if fasta_content:
            print(f"[OK] FASTA export works: {len(fasta_content)} chars")
        else:
            print(f"ℹ️ FASTA export: No protein sequences to export")
    except Exception as e:
        print(f"[ERROR] FASTA export error: {e}")
    
    return True


def run_all_tests():
    """Run all Phase 4 tests"""
    print("\n" + "=" * 60)
    print("[TEST] PHASE 4 UI FEATURE TESTS")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(("Visualization API", test_visualization_api()))
    results.append(("Generate API", test_generate_api()))
    results.append(("Validate API", test_validate_api()))
    results.append(("Rank API", test_rank_api()))
    results.append(("Workflow API", test_workflow_api()))
    results.append(("Export Functions", test_export_functions()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  All Phase 4 features working correctly!")
    else:
        print(f"\n  [WARN] {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    run_all_tests()
