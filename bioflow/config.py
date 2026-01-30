import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = os.getenv("QDRANT_PATH")  # optionnel (si tu utilises Qdrant en mode fichier)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "molecules_v2")

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

# --- Ajouts nécessaires pour server/api.py ---
BEST_MODEL_RUN = os.getenv(
    "BEST_MODEL_RUN",
    str(BASE_DIR / "runs" / "20260127_190751_DAVIS")
)

MODEL_FILENAME = os.getenv("MODEL_FILENAME", "model.pkl")
DEEPPURPOSE_RUN_DIR = os.getenv("DEEPPURPOSE_RUN_DIR", BEST_MODEL_RUN)
