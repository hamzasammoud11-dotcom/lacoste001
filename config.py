from bioflow.config import *

# --- aliases attendus par server/api.py ---
QDRANT_HOST = (QDRANT_URL.split("://",1)[-1].split("/",1)[0].split(":",1)[0]) if "QDRANT_URL" in globals() and QDRANT_URL else "localhost"
QDRANT_PORT = int((QDRANT_URL.split("://",1)[-1].split("/",1)[0].split(":",1)[1]) if ("QDRANT_URL" in globals() and QDRANT_URL and ":" in QDRANT_URL.split("://",1)[-1].split("/",1)[0]) else 6333)

COLLECTION_NAME = globals().get("QDRANT_COLLECTION", "molecules_v2")

MODEL_CONFIG = globals().get("MODEL_CONFIG") or {
  "drug_encoding": "MPNN",
  "target_encoding": "CNN",
  "cls_hidden_dims": [1024, 1024, 512]
}

METRICS = globals().get("METRICS", {"similarity": "cosine"})

VALID_DUMMY_DRUG = globals().get("VALID_DUMMY_DRUG", "CCO")
VALID_DUMMY_TARGET = globals().get("VALID_DUMMY_TARGET", "MKKFFDSRREQGGSGLGSGSSG")
