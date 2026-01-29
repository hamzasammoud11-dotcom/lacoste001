"""
Shared configuration for DeepPurpose + Qdrant Pipeline.
Used by: ingest_qdrant.py, server/api.py
"""
import os

# --- MODEL CONFIG ---
# Best performing run with saved model.pt
BEST_MODEL_RUN = r"runs\20260125_104915_KIBA"

# Encoding config - MUST match what was used during training!
# Verified from config.pkl: Morgan + CNN
MODEL_CONFIG = {
    "drug_encoding": "Morgan",      # Morgan fingerprints
    "target_encoding": "CNN",       # CNN for protein sequences
    "cls_hidden_dims": [1024, 1024, 512],
    "hidden_dim_drug": 128,
    "hidden_dim_protein": 128,
}

# Data source - best CI run for ground truth
PREDICTIONS_SOURCE = r"runs\20260125_080409_BindingDB_Kd\predictions_test.csv"

# --- QDRANT CONFIG ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "bio_discovery"

# --- METRICS (from best runs) ---
METRICS = {
    "KIBA": {"CI": 0.7003, "Pearson": 0.5219, "MSE": 0.0008},
    "BindingDB_Kd": {"CI": 0.8083, "Pearson": 0.7679, "MSE": 0.6668},
    "DAVIS": {"CI": 0.7914, "Pearson": 0.5446, "MSE": 0.4684},
}

# --- VALID SEQUENCES FOR API ---
# Minimal valid sequences for encoding (avoids dummy data issues)
VALID_DUMMY_DRUG = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin SMILES
VALID_DUMMY_TARGET = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH"  # Short valid protein
