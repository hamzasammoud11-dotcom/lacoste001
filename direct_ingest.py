"""
Direct Qdrant Cloud Ingestion for Biological Images
====================================================

Bypasses the API server and uploads directly to Qdrant Cloud.
"""
import os
import sys
import json
import base64
import uuid
from pathlib import Path
from datetime import datetime

# Load environment
from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print(f"Connecting to Qdrant Cloud: {QDRANT_URL[:50]}...")

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Check collections
collections = [c.name for c in client.get_collections().collections]
print(f"Existing collections: {collections}")

# Create biological_images collection if needed
COLLECTION = "biological_images"
VECTOR_DIM = 512

if COLLECTION not in collections:
    print(f"Creating collection '{COLLECTION}'...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
    )
    print(f"✓ Created collection '{COLLECTION}'")
else:
    print(f"✓ Collection '{COLLECTION}' exists")

# Load manifest
MANIFEST_PATH = Path(__file__).parent / "data" / "images" / "biological" / "manifest.json"
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

# Handle both formats: list or dict with 'images' key
if isinstance(manifest, dict):
    images = manifest.get('images', [])
else:
    images = manifest

print(f"Loaded {len(images)} images from manifest")

# We need to generate embeddings - use a simple random vector for now
# or use CLIP via the OBM encoder
import numpy as np

def get_clip_embedding(image_path: str) -> list:
    """Generate CLIP embedding for an image."""
    try:
        # Try to use the actual CLIP model
        from bioflow.plugins.obm_encoder import OBMEncoder
        encoder = OBMEncoder()
        
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        b64 = base64.b64encode(img_bytes).decode()
        data_url = f"data:image/png;base64,{b64}"
        
        result = encoder.encode(data_url, modality="image")
        if isinstance(result, list) and len(result) > 0:
            return result[0].vector
        return result.vector
    except Exception as e:
        print(f"  [WARN] CLIP encoding failed: {e}, using placeholder")
        # Return a random but consistent vector based on filename
        np.random.seed(hash(image_path) % (2**32))
        return np.random.randn(VECTOR_DIM).tolist()

def load_image_as_base64(filepath: str) -> str:
    """Load an image and convert to base64 data URL."""
    with open(filepath, "rb") as f:
        img_bytes = f.read()
    ext = Path(filepath).suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime_type = mime_map.get(ext, "image/png")
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

# Ingest all images
print(f"\nIngesting {len(images)} images to Qdrant Cloud...")
points = []

for i, img_meta in enumerate(images):
    filepath = img_meta.get('file_path')
    if not filepath or not os.path.exists(filepath):
        print(f"[{i+1}] ✗ File not found: {filepath}")
        continue
    
    print(f"[{i+1}/{len(images)}] {img_meta.get('filename', 'unknown')}...", end=" ")
    
    # Get embedding
    vector = get_clip_embedding(filepath)
    
    # Build point
    point_id = str(uuid.uuid4())
    
    payload = {
        "content": img_meta.get("notes", "") or img_meta.get("experiment_type", ""),
        "modality": "image",
        "image_type": img_meta.get("image_type"),
        "experiment_type": img_meta.get("experiment_type"),
        "experiment_id": img_meta.get("experiment_id", ""),
        "outcome": img_meta.get("outcome"),
        "quality_score": img_meta.get("quality_score"),
        "cell_line": img_meta.get("cell_line"),
        "treatment": img_meta.get("treatment"),
        "treatment_target": img_meta.get("treatment_target"),
        "concentration": img_meta.get("concentration"),
        "target_protein": img_meta.get("target_protein"),
        "target_mw": img_meta.get("target_mw"),
        "staining": img_meta.get("staining"),
        "magnification": img_meta.get("magnification"),
        "microscope": img_meta.get("microscope"),
        "cell_count": img_meta.get("cell_count"),
        "gel_percentage": img_meta.get("gel_percentage"),
        "num_lanes": img_meta.get("num_lanes"),
        "protocol": img_meta.get("protocol"),
        "notes": img_meta.get("notes"),
        "experiment_date": img_meta.get("experiment_date"),
        "filename": img_meta.get("filename"),
        "file_path": filepath,
        "source": "bioflow_generated",
        # Store a thumbnail reference instead of full base64
        "image_path": filepath,  # We'll load base64 on demand in the API
    }
    
    points.append(PointStruct(
        id=point_id,
        vector=vector,
        payload=payload
    ))
    
    print("✓")
    
    # Upload in batches of 20
    if len(points) >= 20:
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"  → Uploaded batch of {len(points)} points")
        points = []

# Upload remaining
if points:
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"  → Uploaded final batch of {len(points)} points")

# Verify
info = client.get_collection(COLLECTION)
print(f"\n✓ Collection '{COLLECTION}' now has {info.points_count} points")

print("\nDone! Refresh the UI to see images.")
