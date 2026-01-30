# save as: bioflow/migrate_to_named_vectors.py
# run from repo root:
#   python -m bioflow.migrate_to_named_vectors
import os
import json
from pprint import pprint
from uuid import uuid4

def http_get_json(url: str, timeout: float = 5.0):
    import requests
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def http_put_json(url: str, payload: dict, timeout: float = 10.0):
    import requests
    r = requests.put(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def http_post_json(url: str, payload: dict, timeout: float = 10.0):
    import requests
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def main():
    # Force remote Qdrant
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    os.environ["QDRANT_URL"] = qdrant_url
    os.environ.pop("QDRANT_PATH", None)

    print(f"QDRANT_URL = {qdrant_url}")
    print("QDRANT_PATH = None\n")

    # Import services (works when run as `python -m bioflow...` from repo root)
    from bioflow.api.model_service import get_model_service
    from bioflow.api.qdrant_service import get_qdrant_service

    ms = get_model_service(lazy_load=True)
    qs = get_qdrant_service(model_service=ms, reset=False)

    # --- Decide target collection name (safe: do NOT destroy existing "molecules") ---
    src_collection = "molecules"
    dst_collection = "molecules_v2"

    # --- Read existing collections (HTTP, no pydantic parsing) ---
    cols = http_get_json(f"{qdrant_url}/collections")
    existing = [c["name"] for c in cols.get("result", {}).get("collections", [])]
    print("Collections existantes:", existing)

    # --- Create molecules_v2 with NAMED vectors: molecule + text (both 768) ---
    if dst_collection in existing:
        print(f"\n{dst_collection} existe déjà. On ne la recrée pas.")
    else:
        create_payload = {
            "vectors": {
                "molecule": {"size": 768, "distance": "Cosine"},
                "text": {"size": 768, "distance": "Cosine"},
            }
        }
        print(f"\nCréation de {dst_collection} (named vectors: molecule,text) ...")
        res = http_put_json(f"{qdrant_url}/collections/{dst_collection}", create_payload)
        print("Create response status:", res.get("status", "unknown"))

    # --- Minimal test ingest: 1 point with molecule vector, and optional text vector ---
    # We ingest via QdrantService.ingest (so payload matches your app conventions),
    # then we PATCH vectors via raw HTTP upsert to ensure named vectors exist.
    # Reason: your current ingest pathway stores DEFAULT vector; we want named.
    smiles = "CCO"
    meta = {"source": "manual_test", "name": "Ethanol", "smiles": smiles}

    # Build embeddings
       # Build embeddings (ModelService returns EncodingResult, not raw list)
    mol_res = ms.encode_molecule(smiles)
    txt_res = ms.encode_text("Ethanol")

    # try the common field names used in repos like this
    def as_list(encoding_result):
        for attr in ("vector", "embedding", "embeddings"):
            v = getattr(encoding_result, attr, None)
            if v is not None:
                # v can be numpy array / torch tensor / list
                if hasattr(v, "tolist"):
                    return v.tolist()
                return list(v)
        # sometimes it's a dict like {"vector": [...]}
        if isinstance(encoding_result, dict):
            v = encoding_result.get("vector") or encoding_result.get("embedding")
            if v is not None:
                if hasattr(v, "tolist"):
                    return v.tolist()
                return list(v)
        raise TypeError(f"Cannot extract vector from type={type(encoding_result)}; attrs={dir(encoding_result)}")

    mol_vec = as_list(mol_res)
    text_vec = as_list(txt_res)

    point_id = str(uuid4())

    upsert_payload = {
        "points": [
            {
                "id": point_id,
                "payload": {
                    "content": smiles,
                    "modality": "molecule",
                    **meta,
                },
                "vector": {
                    "molecule": mol_vec,
                    "text": text_vec,
                },
            }
        ]
    }


    # Upsert with named vectors via raw HTTP (bypasses qdrant-client schema issues)
    upsert_payload = {
        "points": [
            {
                "id": point_id,
                "payload": {
                    "content": smiles,
                    "modality": "molecule",
                    **meta,
                },
                "vector": {
                    "molecule": mol_vec.tolist() if hasattr(mol_vec, "tolist") else list(mol_vec),
                    "text": text_vec.tolist() if hasattr(text_vec, "tolist") else list(text_vec),
                },
            }
        ]
    }

    print(f"\nUpsert test point -> {dst_collection} (id={point_id}) ...")
    upsert_res = http_put_json(
        f"{qdrant_url}/collections/{dst_collection}/points?wait=true",
        upsert_payload,
        timeout=30.0,
    )
    print("Upsert response status:", upsert_res.get("status", "unknown"))

    # --- Inspect collection schema and a sample point (HTTP scroll) ---
    print(f"\nInspect {dst_collection} ...")
    info = http_get_json(f"{qdrant_url}/collections/{dst_collection}")
    result = info.get("result", {})
    print("points_count:", result.get("points_count"))
    print("vectors schema:")
    pprint(result.get("config", {}).get("params", {}).get("vectors"))

    scroll_payload = {"limit": 3, "with_payload": True, "with_vector": True}
    scroll = http_post_json(f"{qdrant_url}/collections/{dst_collection}/points/scroll", scroll_payload, timeout=15.0)
    pts = scroll.get("result", {}).get("points", [])

    if not pts:
        print("Aucun point trouvé (unexpected).")
        return

    p0 = pts[0]
    payload = p0.get("payload", {}) or {}
    vectors = p0.get("vector", None)

    print("\nSample point id:", p0.get("id"))
    print("Payload keys:", sorted(payload.keys()))
    print("Payload preview:", json.dumps(payload, indent=2)[:600])

    if isinstance(vectors, dict):
        print("\nVectors: NAMED")
        print("Named vector keys:", list(vectors.keys()))
        for k in list(vectors.keys())[:2]:
            v = vectors.get(k)
            print(f"dim({k}) =", len(v) if v else 0)
    else:
        print("\nVectors: DEFAULT (unexpected here)")
        print("Default vector dim:", len(vectors) if vectors else 0)

    print("\nDONE.")

if __name__ == "__main__":
    main()
