#!/usr/bin/env python3
"""Test the Enhanced Search API endpoint."""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    r = requests.get(f"{BASE_URL}/health")
    print(f"Health: {r.status_code}")
    return r.status_code == 200

def test_enhanced_search():
    """Test enhanced search endpoint."""
    data = {
        "query": "BRCA1 breast cancer",
        "top_k": 3,
        "use_mmr": True,
    }
    print(f"\n📡 Testing /api/search with: {data}")
    
    r = requests.post(f"{BASE_URL}/api/search", json=data)
    print(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        result = r.json()
        print(f"✅ Results: {result.get('returned')} of {result.get('total_found')}")
        print(f"   Diversity: {result.get('diversity_score'):.4f}")
        print(f"   Search time: {result.get('search_time_ms'):.1f}ms")
        
        for item in result.get('results', [])[:3]:
            content = item.get('content', '')[:60]
            print(f"   [{item.get('rank')}] score={item.get('score'):.3f} - {content}...")
        return True
    else:
        print(f"❌ Error: {r.text}")
        return False

def test_hybrid_search():
    """Test hybrid search endpoint."""
    data = {
        "query": "kinase inhibitor",
        "keywords": ["cancer", "therapy"],
        "top_k": 3,
    }
    print(f"\n📡 Testing /api/search/hybrid with: {data}")
    
    r = requests.post(f"{BASE_URL}/api/search/hybrid", json=data)
    print(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        result = r.json()
        print(f"✅ Results: {result.get('returned')} of {result.get('total_found')}")
        return True
    else:
        print(f"❌ Error: {r.text}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("BioFlow Enhanced Search API Test")
    print("=" * 50)
    
    # Wait for server to be ready
    for i in range(10):
        try:
            if test_health():
                break
        except requests.exceptions.ConnectionError:
            print(f"Waiting for server... ({i+1}/10)")
            time.sleep(1)
    else:
        print("❌ Server not available")
        exit(1)
    
    # Run tests
    test_enhanced_search()
    test_hybrid_search()
    
    print("\n" + "=" * 50)
    print("Tests complete!")
