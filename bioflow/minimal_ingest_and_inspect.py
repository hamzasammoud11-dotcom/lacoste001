# save as: minimal_ingest_and_inspect.py
# run from repo root: python -m bioflow.minimal_ingest_and_inspect
# (or: python minimal_ingest_and_inspect.py if bioflow is importable from cwd)

import os
import json
import uuid
import requests
from pprint import pprint
def get_stats_http(base_url: str, collection: str):
    r = requests.get(f"{base_url}/collections/{collection}")
    r.raise_for_status()
    return r.json()

def main():
    # 1) Force Qdrant remote (so we don't accidentally write to ./qdrant_data)
    os.environ["QDRANT_URL"] = os.environ.get("QDRANT_URL", "http://localhost:6333")
    os.environ.pop("QDRANT_PATH", None)

    print(f"QDRANT_URL = {os.environ['QDRANT_URL']}")
    print("QDRANT_PATH = None\n")

    # 2) Import your services (expects you run from repo root)
    from bioflow.api.model_service import get_model_service
    from bioflow.api.qdrant_service import get_qdrant_service

    ms = get_model_service(lazy_load=True)
    qs = get_qdrant_service(model_service=ms, reset=True)

    # 3) Minimal ingest: 1 molecule point into collection "molecules"
    smiles = "CCO"
    meta = {
        "source": "manual_test",
        "name": "Ethanol",
        "smiles": smiles,
    }

    # Qdrant point id must be UUID or unsigned int. Make it stable and valid:
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "manual:ethanol"))

    res = qs.ingest(
        content=smiles,
        modality="molecule",
        metadata=meta,
        collection="molecules",
        id=point_id,  # <-- FIXED (valid UUID)
    )
    print("IngestResult:")
    print(res)
    print()

    # 4) Inspect: collections + stats + sample payload keys via scroll
    cols = qs.list_collections()
    print("Collections:", cols)
    print()

    client = qs._get_client()

    for c in cols:
        print("=" * 80)
        print("COLLECTION:", c)
        print("=" * 80)
        try:
            stats = get_stats_http(os.environ["QDRANT_URL"], c)
            print("Stats (HTTP) points_count:", stats["result"].get("points_count"))
            print("Stats (HTTP) vectors_count:", stats["result"].get("vectors_count"))
            # schema des vecteurs (utile)
            print("Vectors schema:", stats["result"]["config"]["params"]["vectors"])
        except Exception as e:
            print("Stats (HTTP) error:", e)

        # scroll 5 points
        try:
            points, _ = client.scroll(
                collection_name=c,
                limit=5,
                with_payload=True,
                with_vectors=True,
            )
        except TypeError:
            points, _ = client.scroll(
                collection_name=c,
                limit=5,
                with_payload=True,
                with_vectors=True,
                scroll_filter=None,
            )

        if not points:
            print("Empty collection (no points).")
            continue

        p0 = points[0]
        payload = p0.payload or {}
        print("\nSample point id:", p0.id)
        print("Payload keys:", sorted(list(payload.keys())))
        print("Payload preview:", json.dumps(payload, indent=2)[:800])

        vec = getattr(p0, "vector", None)
        if isinstance(vec, dict):
            print("\nVectors: NAMED")
            print("Named vector keys:", list(vec.keys()))
            first_key = next(iter(vec.keys()))
            print("Example named vector dim:", len(vec[first_key]) if vec[first_key] else 0)
        else:
            print("\nVectors: DEFAULT")
            print("Default vector dim:", len(vec) if vec else 0)

        print()

    print("\nDONE.")

if __name__ == "__main__":
    main()
