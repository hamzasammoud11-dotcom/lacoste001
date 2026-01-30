import os, sys, json, re
from collections import Counter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

def is_chembl_payload(payload: dict) -> bool:
    if not payload:
        return False
    src = payload.get("source")
    if isinstance(src, str) and src.lower() == "chembl":
        return True
    if payload.get("chembl_id"):
        return True
    url = payload.get("url", "")
    if isinstance(url, str) and "ebi.ac.uk/chembl" in url.lower():
        return True
    cid = payload.get("id", "") or payload.get("record_id", "")
    if isinstance(cid, str) and cid.lower().startswith("chembl:"):
        return True
    return False

def main():
    print("QDRANT_URL =", os.getenv("QDRANT_URL"))
    print("QDRANT_PATH =", os.getenv("QDRANT_PATH"))

    from bioflow.api.qdrant_service import get_qdrant_service
    q = get_qdrant_service(reset=True)
    client = q._get_client()

    cols = q.list_collections()
    print("\nCollections =", cols)

    if not cols:
        print("No collections found.")
        return

    for cname in cols:
        print("\n================================================================================")
        print("COLLECTION:", cname)
        print("================================================================================")

        # 1) Scroll sample points (payload + vectors)
        try:
            pts, _ = client.scroll(
                collection_name=cname,
                limit=20,
                with_payload=True,
                with_vectors=True,
            )
        except TypeError:
            pts, _ = client.scroll(
                cname,
                limit=20,
                with_payload=True,
                with_vectors=True,
            )

        if not pts:
            print("Empty collection (no points).")
            continue

        # Determine vector schema from first point
        first_vec = getattr(pts[0], "vector", None)
        if isinstance(first_vec, dict):
            print("Vector schema: NAMED")
            for k, v in first_vec.items():
                dim = len(v) if v is not None else None
                print(f" - key={k} dim={dim}")
        else:
            dim = len(first_vec) if first_vec is not None else None
            print("Vector schema: DEFAULT")
            print(" - dim =", dim)

        # Payload keys statistics + chembl presence
        key_counter = Counter()
        chembl_hits = 0
        modality_counter = Counter()
        source_counter = Counter()

        for p in pts:
            payload = p.payload or {}
            key_counter.update(payload.keys())

            mod = payload.get("modality")
            if isinstance(mod, str):
                modality_counter[mod] += 1

            src = payload.get("source")
            if isinstance(src, str):
                source_counter[src] += 1

            if is_chembl_payload(payload):
                chembl_hits += 1

        common_keys = [k for k, _ in key_counter.most_common(30)]
        print("\nTop payload keys (sampled 20):")
        print(common_keys)

        print("\nSample modality counts (20):", dict(modality_counter))
        print("Sample source counts (20):", dict(source_counter))
        print(f"ChEMBL-like payloads in sample: {chembl_hits}/20")

        # Show 1 example payload (trim large fields)
        sample_payload = pts[0].payload or {}
        trimmed = {}
        for k, v in sample_payload.items():
            s = v
            if isinstance(v, (list, dict)):
                s = str(v)
            if isinstance(s, str) and len(s) > 200:
                s = s[:200] + "..."
            trimmed[k] = s

        print("\nExample payload (trimmed):")
        print(json.dumps(trimmed, indent=2, ensure_ascii=False))

        # 2) Raw HTTP schema (avoids pydantic parsing)
        try:
            import requests
            base = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
            r = requests.get(f"{base}/collections/{cname}", timeout=10)
            print("\nRaw HTTP /collections/{name} status:", r.status_code)
            if r.ok:
                data = r.json()
                # Extract vectors config if present
                cfg = (((data.get("result") or {}).get("config") or {}).get("params") or {}).get("vectors")
                print("Raw vectors config type:", "dict(named)" if isinstance(cfg, dict) else type(cfg).__name__)
        except Exception as e:
            print("\nRaw HTTP schema fetch failed:", e)

if __name__ == "__main__":
    main()
