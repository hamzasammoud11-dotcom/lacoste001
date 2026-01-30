"""
Ingest Biomedical Images into BioFlow Qdrant
=============================================

Ingests the downloaded biomedical images into the vector database.
"""

import os
import sys
import json
import base64
import requests
from pathlib import Path

# API endpoint
API_BASE = "http://localhost:8000"
IMAGES_DIR = Path(__file__).parent / "data" / "images"
MANIFEST_PATH = IMAGES_DIR / "manifest.json"


def load_image_as_base64(filepath: str) -> str:
    """Load an image and convert to base64."""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ingest_single_image(image_info: dict) -> bool:
    """Ingest a single image via the API."""
    try:
        filepath = image_info["image"]
        
        # Load image as base64
        img_base64 = load_image_as_base64(filepath)
        
        # Prepare request payload
        payload = {
            "image": f"data:image/png;base64,{img_base64}",
            "image_type": image_info.get("image_type", "other"),
            "description": image_info.get("description", ""),
            "caption": image_info.get("description", ""),
            "experiment_id": image_info.get("metadata", {}).get("compound_id", ""),
            "metadata": image_info.get("metadata", {}),
            "collection": "bioflow_memory"
        }
        
        response = requests.post(
            f"{API_BASE}/api/ingest/image",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✓ Ingested: {os.path.basename(filepath)}")
            return True
        else:
            print(f"  ✗ Failed: {os.path.basename(filepath)} - {response.status_code}: {response.text[:100]}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {os.path.basename(filepath)} - {e}")
        return False


def main():
    print("=" * 60)
    print("BioFlow Image Ingestion")
    print("=" * 60)
    
    # Check if manifest exists
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        print("Run download_biomedical_images.py first!")
        return
    
    # Load manifest
    with open(MANIFEST_PATH, "r") as f:
        images = json.load(f)
    
    print(f"\nImages to ingest: {len(images)}")
    print(f"API endpoint: {API_BASE}")
    print()
    
    # Check API health
    try:
        health = requests.get(f'{API_BASE}/health', timeout=15)
        if health.status_code != 200:
            print("ERROR: API not healthy")
            return
        print("✓ API is healthy\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to API - {e}")
        print("Make sure the server is running: python start_server.py")
        return
    
    successful = 0
    failed = 0
    
    for i, img in enumerate(images, 1):
        # Get the image filename and resolve full path
        image_filename = img.get("image", "")
        
        # Try multiple paths to find the image
        possible_paths = [
            IMAGES_DIR / image_filename,  # data/images/filename.png
            Path(image_filename),  # As-is (might be absolute)
            IMAGES_DIR.parent / image_filename,  # data/filename.png
        ]
        
        filepath = None
        for p in possible_paths:
            if p.exists():
                filepath = str(p)
                break
        
        if not filepath:
            print(f"[{i}/{len(images)}] Skipping (file not found): {image_filename}")
            failed += 1
            continue
        
        # Update img dict with resolved path
        img["image"] = filepath
            
        print(f"[{i}/{len(images)}] Ingesting: {os.path.basename(filepath)}")
        if ingest_single_image(img):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Ingestion complete: {successful} successful, {failed} failed")
    print("=" * 60)
    
    # Verify in database
    print("\nVerifying images in database...")
    try:
        # Search for an ingested image
        verify_payload = {
            "query": "kinase structure protein",
            "modality": "image",
            "top_k": 5,
            "use_mmr": False
        }
        verify_resp = requests.post(f"{API_BASE}/api/search", json=verify_payload, timeout=30)
        if verify_resp.status_code == 200:
            results = verify_resp.json()
            print(f"✓ Found {len(results.get('results', []))} images in search results")
        else:
            print(f"✗ Verification search failed: {verify_resp.status_code}")
    except Exception as e:
        print(f"✗ Verification failed: {e}")


if __name__ == "__main__":
    main()
