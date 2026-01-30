# save as: bioflow/cross_modal.py
# run (from repo root): python -m bioflow.cross_modal
import os
from pprint import pprint

COLLECTION = "molecules_v2"

def main():
    # Force remote Qdrant
    os.environ["QDRANT_URL"] = os.environ.get("QDRANT_URL", "http://localhost:6333")
    os.environ.pop("QDRANT_PATH", None)

    print(f"QDRANT_URL = {os.environ['QDRANT_URL']}")
    print("QDRANT_PATH = None\n")

    from bioflow.api.model_service import get_model_service
    from bioflow.api.qdrant_service import get_qdrant_service

    ms = get_model_service(lazy_load=True)
    qs = get_qdrant_service(model_service=ms, reset=False)
    client = qs._get_client()

    # ---------
    # 1) TEXT -> search on named vector "text"
    # ---------
    text_query = "EGFR inhibitor"
    text_res = ms.encode_text(text_query)

    def as_list(res):
        for attr in ("vector", "embedding", "embeddings"):
            v = getattr(res, attr, None)
            if v is not None:
                return v.tolist() if hasattr(v, "tolist") else list(v)
        if isinstance(res, dict):
            v = res.get("vector") or res.get("embedding")
            if v is not None:
                return v.tolist() if hasattr(v, "tolist") else list(v)
        raise TypeError(f"Cannot extract vector from {type(res)}")

    q_text = as_list(text_res)

    print("=" * 80)
    print("TEXT -> MOLECULES")
    print("=" * 80)
    hits_text = client.search(
        collection_name=COLLECTION,
        query_vector=("text", q_text),   # <-- named vector
        limit=5,
        with_payload=True,
    )
    for i, h in enumerate(hits_text, 1):
        payload = h.payload or {}
        print(f"[{i}] score={h.score:.4f} id={h.id}")
        print("    name  :", payload.get("name"))
        print("    smiles:", payload.get("smiles"))
        print("    source:", payload.get("source"))
        print("    modality:", payload.get("modality"))
        print()

    # ---------
    # 2) SMILES -> search on named vector "molecule"
    # ---------
    smiles_query = "CCO"
    mol_res = ms.encode_molecule(smiles_query)
    q_mol = as_list(mol_res)

    print("=" * 80)
    print("SMILES -> MOLECULES")
    print("=" * 80)
    hits_mol = client.search(
        collection_name=COLLECTION,
        query_vector=("molecule", q_mol),  # <-- named vector
        limit=5,
        with_payload=True,
    )
    for i, h in enumerate(hits_mol, 1):
        payload = h.payload or {}
        print(f"[{i}] score={h.score:.4f} id={h.id}")
        print("    name  :", payload.get("name"))
        print("    smiles:", payload.get("smiles"))
        print("    source:", payload.get("source"))
        print("    modality:", payload.get("modality"))
        print()

    print("DONE.")

if __name__ == "__main__":
    main()
