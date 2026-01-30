# bioflow/ingest_chembl_to_molecules_v2.py
# Run (from repo root): python -m bioflow.ingest_chembl_to_molecules_v2 --file PATH --format auto
# Env: QDRANT_URL=http://localhost:6333 (recommended)

import os
import re
import csv
import json
import uuid
import argparse
from typing import Dict, Any, Iterable, List, Optional, Tuple

def _norm(s: Optional[str]) -> str:
    return (s or "").strip()

def _guess_format(path: str) -> str:
    p = path.lower()
    if p.endswith(".jsonl"):
        return "jsonl"
    if p.endswith(".json"):
        return "json"
    if p.endswith(".tsv"):
        return "tsv"
    if p.endswith(".csv"):
        return "csv"
    return "csv"

def _iter_rows_csv_tsv(path: str, delim: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            yield {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}

def _iter_rows_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _iter_rows_json(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                yield obj
    elif isinstance(data, dict):
        # allow {"items":[...]}
        items = data.get("items") or data.get("data") or []
        if isinstance(items, list):
            for obj in items:
                if isinstance(obj, dict):
                    yield obj

def _pick(row: Dict[str, Any], keys: List[str]) -> Optional[str]:
    # case-insensitive key lookup
    lower = {k.lower(): k for k in row.keys()}
    for k in keys:
        kk = k.lower()
        if kk in lower:
            v = row.get(lower[kk])
            if v is None:
                continue
            v = str(v).strip()
            if v:
                return v
    return None

def _extract_embedding(obj: Any) -> List[float]:
    """
    Encoders sometimes return EncodingResult objects.
    We try common attribute names safely.
    """
    # already list/tuple
    if isinstance(obj, (list, tuple)):
        return [float(x) for x in obj]

    # numpy array / torch tensor
    for attr in ("tolist",):
        if hasattr(obj, attr):
            try:
                out = getattr(obj, attr)()
                if isinstance(out, (list, tuple)):
                    return [float(x) for x in out]
            except Exception:
                pass

    # EncodingResult patterns
    for attr in ("vector", "embedding", "embeddings", "values", "data"):
        if hasattr(obj, attr):
            try:
                v = getattr(obj, attr)
                if callable(v):
                    v = v()
                return _extract_embedding(v)
            except Exception:
                pass

    raise TypeError(f"Cannot extract embedding from type: {type(obj)}")

def _ensure_collection_named_vectors(client, name: str, dim: int = 768) -> None:
    # Create collection if missing. If exists, do nothing.
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        return

    from qdrant_client.models import VectorParams, Distance, VectorsConfig

    client.create_collection(
        collection_name=name,
        vectors_config=VectorsConfig(
            vectors={
                "molecule": VectorParams(size=dim, distance=Distance.COSINE),
                "text": VectorParams(size=dim, distance=Distance.COSINE),
            }
        ),
    )

def _batched(it: Iterable[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to ChEMBL export (csv/tsv/json/jsonl)")
    parser.add_argument("--format", default="auto", choices=["auto", "csv", "tsv", "json", "jsonl"])
    parser.add_argument("--collection", default="molecules_v2")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = parser.parse_args()

    os.environ["QDRANT_URL"] = os.environ.get("QDRANT_URL", "http://localhost:6333")
    os.environ.pop("QDRANT_PATH", None)

    print(f"QDRANT_URL = {os.environ['QDRANT_URL']}")
    print("QDRANT_PATH = None\n")

    path = args.file
    fmt = args.format
    if fmt == "auto":
        fmt = _guess_format(path)

    # --- Load services
    from bioflow.api.model_service import get_model_service
    from bioflow.api.qdrant_service import get_qdrant_service

    ms = get_model_service(lazy_load=True)
    qs = get_qdrant_service(model_service=ms, reset=False)
    client = qs._get_client()

    # --- Ensure collection
    _ensure_collection_named_vectors(client, args.collection, dim=768)
    print(f"Collection ready: {args.collection}\n")

    # --- Row iterator
    if fmt == "csv":
        rows = _iter_rows_csv_tsv(path, ",")
    elif fmt == "tsv":
        rows = _iter_rows_csv_tsv(path, "\t")
    elif fmt == "jsonl":
        rows = _iter_rows_jsonl(path)
    elif fmt == "json":
        rows = _iter_rows_json(path)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    # --- Ingest loop
    from qdrant_client.models import PointStruct

    total = 0
    inserted = 0
    skipped = 0

    def make_id(chembl_id: str, smiles: str) -> str:
        # deterministic UUID (stable across reruns)
        key = f"chembl:{chembl_id}|{smiles}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    # Common field name variants you may have in exports
    KEYS_SMILES = ["smiles", "canonical_smiles", "molecule_smiles"]
    KEYS_NAME   = ["name", "pref_name", "molecule_name", "compound_name"]
    KEYS_ID     = ["chembl_id", "molecule_chembl_id", "id"]

    for batch_rows in _batched(rows, args.batch):
        points: List[PointStruct] = []

        for row in batch_rows:
            if args.limit and total >= args.limit:
                break
            total += 1

            smiles = _pick(row, KEYS_SMILES)
            chembl_id = _pick(row, KEYS_ID) or "UNKNOWN"
            name = _pick(row, KEYS_NAME) or ""

            if not smiles:
                skipped += 1
                continue

            try:
                # Encodings (IMPORTANT: extract actual vectors)
                mol_res = ms.encode_molecule(smiles)
                txt_res = ms.encode_text(name if name else smiles)

                mol_vec = _extract_embedding(mol_res)
                txt_vec = _extract_embedding(txt_res)

                pid = make_id(chembl_id, smiles)

                payload = {
                    "content": smiles,
                    "modality": "molecule",
                    "source": "chembl",
                    "smiles": smiles,
                    "name": name,
                    "chembl_id": chembl_id,
                }

                points.append(
                    PointStruct(
                        id=pid,
                        vector={"molecule": mol_vec, "text": txt_vec},
                        payload=payload,
                    )
                )
            except Exception:
                skipped += 1
                continue

        if points:
            client.upsert(collection_name=args.collection, points=points, wait=True)
            inserted += len(points)

        if args.limit and total >= args.limit:
            break

        if total % (args.batch * 10) == 0:
            print(f"Progress: total={total} inserted={inserted} skipped={skipped}")

    print("\nDONE")
    print(f"total={total} inserted={inserted} skipped={skipped}")
    print(f"collection={args.collection}")

if __name__ == "__main__":
    main()
