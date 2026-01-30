"""
Ingest Biological Images into Qdrant
=====================================

Processes biological images (Western blots, gels, microscopy) and:
1. Generates CLIP embeddings for each image
2. Stores in Qdrant 'biological_images' collection
3. Indexes metadata for faceted filtering

Usage:
    python ingest_biological_images.py [--api-url http://localhost:8000]
"""

import os
import sys
import json
import base64
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid

# API endpoint
DEFAULT_API_URL = "http://localhost:8000"

# Data directory
DATA_DIR = Path(__file__).parent / "data" / "images" / "biological"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# Collection name for biological images
COLLECTION_NAME = "biological_images"


def load_image_as_base64(filepath: str) -> str:
    """Load an image and convert to base64 data URL."""
    with open(filepath, "rb") as f:
        img_bytes = f.read()
    
    # Detect format
    ext = Path(filepath).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/png")
    
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def create_biological_images_collection(api_url: str) -> bool:
    """
    Ensure the biological_images collection exists in Qdrant.
    Uses the existing /api/ingest infrastructure to create via first insert.
    """
    try:
        # Check collections
        resp = requests.get(f"{api_url}/api/collections", timeout=10)
        if resp.status_code == 200:
            collections = resp.json().get("collections", [])
            if COLLECTION_NAME in collections:
                print(f"✓ Collection '{COLLECTION_NAME}' already exists")
                return True
        
        print(f"Collection '{COLLECTION_NAME}' will be created on first ingest")
        return True
    except Exception as e:
        print(f"✗ Error checking collections: {e}")
        return False


def ingest_single_image(api_url: str, image_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ingest a single biological image via the API.
    
    Returns:
        {"success": bool, "id": str, "error": str or None}
    """
    filepath = image_meta.get("file_path")
    if not filepath or not os.path.exists(filepath):
        return {
            "success": False,
            "id": None,
            "error": f"File not found: {filepath}"
        }
    
    try:
        # Load image
        image_b64 = load_image_as_base64(filepath)
        
        # Generate unique ID
        point_id = str(uuid.uuid4())
        
        # Map image_type to standardized format
        img_type = image_meta.get("image_type", "other")
        if img_type == "western_blot":
            img_type = "gel"  # Store under 'gel' for filtering
        
        # Build payload for /api/ingest/image endpoint
        payload = {
            "image": image_b64,
            "image_type": img_type,
            "description": image_meta.get("notes", "") or image_meta.get("experiment_type", "Biological image"),
            "caption": f"{image_meta.get('experiment_type', '')} - {image_meta.get('cell_line', '')} - {image_meta.get('treatment', '')}",
            "experiment_id": image_meta.get("experiment_id", ""),
            "collection": COLLECTION_NAME,  # Use dedicated collection
            "metadata": {
                # Core searchable fields
                "image_type": image_meta.get("image_type"),
                "experiment_type": image_meta.get("experiment_type"),
                "experiment_id": image_meta.get("experiment_id"),
                "outcome": image_meta.get("outcome"),
                "quality_score": image_meta.get("quality_score"),
                
                # Cell/treatment info
                "cell_line": image_meta.get("cell_line"),
                "treatment": image_meta.get("treatment"),
                "treatment_target": image_meta.get("treatment_target"),
                "concentration": image_meta.get("concentration"),
                "conditions": image_meta.get("conditions", {}),
                
                # Protein/target info  
                "target_protein": image_meta.get("target_protein"),
                "target_mw": image_meta.get("target_mw"),
                "target_function": image_meta.get("target_function"),
                "protein": image_meta.get("protein"),
                
                # Microscopy-specific
                "staining": image_meta.get("staining"),
                "staining_purpose": image_meta.get("staining_purpose"),
                "channels": image_meta.get("channels", []),
                "magnification": image_meta.get("magnification"),
                "microscope": image_meta.get("microscope"),
                "imaging_mode": image_meta.get("imaging_mode"),
                "cell_count": image_meta.get("cell_count"),
                
                # Gel-specific
                "gel_percentage": image_meta.get("gel_percentage"),
                "stain": image_meta.get("stain"),
                "expression_system": image_meta.get("expression_system"),
                "purpose": image_meta.get("purpose"),
                "num_lanes": image_meta.get("num_lanes"),
                
                # Protocol & notes
                "protocol": image_meta.get("protocol"),
                "notes": image_meta.get("notes"),
                
                # Dates
                "experiment_date": image_meta.get("experiment_date"),
                "acquisition_date": image_meta.get("acquisition_date"),
                
                # File reference
                "filename": image_meta.get("filename"),
                "file_path": filepath,
                "source": "bioflow_generated",
                
                # For UI display
                "image": image_b64,  # Include base64 for gallery display
            }
        }
        
        # Send to API
        resp = requests.post(
            f"{api_url}/api/ingest/image",
            json=payload,
            timeout=60
        )
        
        if resp.status_code == 200:
            result = resp.json()
            return {
                "success": True,
                "id": result.get("id", point_id),
                "error": None
            }
        else:
            return {
                "success": False,
                "id": None,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}"
            }
    
    except Exception as e:
        return {
            "success": False,
            "id": None,
            "error": str(e)
        }


def ingest_all_images(api_url: str, manifest: List[Dict[str, Any]], batch_size: int = 5) -> Dict[str, int]:
    """
    Ingest all images from manifest.
    
    Returns:
        {"successful": int, "failed": int, "total": int}
    """
    total = len(manifest)
    successful = 0
    failed = 0
    
    print(f"\nIngesting {total} biological images to Qdrant...")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"API: {api_url}")
    print()
    
    for i, img_meta in enumerate(manifest, 1):
        filename = img_meta.get("filename", "unknown")
        img_type = img_meta.get("image_type", "unknown")
        
        print(f"[{i}/{total}] {img_type}: {filename}...", end=" ")
        
        result = ingest_single_image(api_url, img_meta)
        
        if result["success"]:
            print("✓")
            successful += 1
        else:
            print(f"✗ ({result['error'][:50]}...)" if result['error'] else "✗")
            failed += 1
    
    return {
        "successful": successful,
        "failed": failed,
        "total": total
    }


def verify_ingestion(api_url: str) -> Dict[str, Any]:
    """
    Verify images were ingested by running test searches.
    """
    print("\n" + "="*60)
    print("Verifying ingestion...")
    
    results = {}
    
    # Test 1: Count images by type
    try:
        # Search for each type
        for img_type in ["gel", "western_blot", "microscopy"]:
            resp = requests.post(
                f"{api_url}/api/search",
                json={
                    "query": img_type,
                    "modality": "image",
                    "top_k": 5,
                    "collection": COLLECTION_NAME,
                    "use_mmr": False
                },
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                count = data.get("returned", 0)
                results[img_type] = count
                print(f"  ✓ Found {count} results for '{img_type}'")
            else:
                results[img_type] = 0
                print(f"  ✗ Search for '{img_type}' failed: {resp.status_code}")
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Ingest biological images into Qdrant")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API base URL")
    parser.add_argument("--manifest", type=str, help="Path to manifest.json")
    parser.add_argument("--generate", action="store_true", help="Generate images first if not exists")
    args = parser.parse_args()
    
    print("="*60)
    print("BioFlow Biological Image Ingestion")
    print("="*60)
    
    # Determine manifest path
    manifest_path = Path(args.manifest) if args.manifest else MANIFEST_PATH
    
    # Generate images if requested or if manifest doesn't exist
    if args.generate or not manifest_path.exists():
        print("\n📷 Generating biological images first...")
        try:
            from generate_biological_images import generate_dataset
            generate_dataset()
        except ImportError:
            print("ERROR: generate_biological_images.py not found!")
            print("Please run that script first to create the images.")
            sys.exit(1)
    
    # Check manifest exists
    if not manifest_path.exists():
        print(f"\nERROR: Manifest not found at {manifest_path}")
        print("Run generate_biological_images.py first!")
        sys.exit(1)
    
    # Load manifest
    print(f"\n📄 Loading manifest from {manifest_path}...")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    print(f"   Found {len(manifest)} images in manifest")
    
    # Check API health
    print(f"\n🔌 Checking API at {args.api_url}...")
    try:
        health = requests.get(f"{args.api_url}/health", timeout=10)
        if health.status_code != 200:
            print(f"ERROR: API not healthy (HTTP {health.status_code})")
            sys.exit(1)
        print("   ✓ API is healthy")
    except Exception as e:
        print(f"ERROR: Cannot connect to API - {e}")
        print("Make sure the server is running: python start_server.py")
        sys.exit(1)
    
    # Create collection if needed
    create_biological_images_collection(args.api_url)
    
    # Ingest images
    results = ingest_all_images(args.api_url, manifest)
    
    # Summary
    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)
    print(f"Total images:     {results['total']}")
    print(f"Successful:       {results['successful']}")
    print(f"Failed:           {results['failed']}")
    print(f"Success rate:     {results['successful']/results['total']*100:.1f}%")
    
    # Verify
    if results["successful"] > 0:
        verify_results = verify_ingestion(args.api_url)
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. Open the UI: http://localhost:3000")
        print("2. Go to Discovery page")
        print("3. Click 'Gels & Microscopy' tab")
        print("4. You should now see images!")
        print("5. Upload your own image to find similar experiments")
    
    print("="*60)


if __name__ == "__main__":
    main()
