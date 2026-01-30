"""Check what's in the biological_images collection."""
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv('QDRANT_URL')
key = os.getenv('QDRANT_API_KEY')
print(f"Connecting to: {url[:50]}...")

client = QdrantClient(url=url, api_key=key)

# Get collection info
info = client.get_collection('biological_images')
print(f"Collection has {info.points_count} points")

# Get sample records
result = client.scroll('biological_images', limit=10, with_payload=True)
print(f"\nSample records:")

for p in result[0]:
    payload = p.payload
    print(f"  - ID: {str(p.id)[:8]}")
    print(f"    image_type: {payload.get('image_type')}")
    print(f"    experiment_type: {payload.get('experiment_type')}")
    print(f"    cell_line: {payload.get('cell_line')}")
    print(f"    outcome: {payload.get('outcome')}")
    print()
