"""Check database contents"""
from qdrant_client import QdrantClient

client = QdrantClient(path='./qdrant_data')

# Get collections
colls = client.get_collections().collections
print('Collections:', [c.name for c in colls])

for c in colls:
    count = client.count(c.name).count
    print(f'\n{c.name}: {count} points')
    
    # Sample some points
    results = client.scroll(collection_name=c.name, limit=5, with_payload=True)
    for r in results[0]:
        modality = r.payload.get('modality', 'NONE')
        content = str(r.payload.get('content', ''))[:50]
        print(f"  - {r.id}: modality={modality}, content={content}...")

# Check for any image modality
print('\n--- Checking for image modality ---')
from qdrant_client.models import Filter, FieldCondition, MatchValue
for c in colls:
    try:
        results = client.scroll(
            collection_name=c.name,
            limit=10,
            with_payload=True,
            scroll_filter=Filter(
                must=[FieldCondition(key="modality", match=MatchValue(value="image"))]
            )
        )
        print(f"{c.name}: {len(results[0])} images found")
    except Exception as e:
        print(f"{c.name}: error - {e}")
