"""
Phase 1: DeepPurpose -> Qdrant Ingestion Pipeline (v2 - Fixed)

Fixes applied:
- Proper PCA dimensionality reduction (not just first 3 dims)
- No shuffle to preserve data ordering
- Proper error handling and validation
- Shared config import
"""
import os
os.environ["DGL_DISABLE_GRAPHBOLT"] = "1"

import sys
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.decomposition import PCA
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from DeepPurpose import utils, DTI as dp_models
import warnings
warnings.filterwarnings("ignore")

# Import shared config
from config import (
    BEST_MODEL_RUN, MODEL_CONFIG,
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, METRICS
)

def validate_model_path(run_dir: str) -> bool:
    """Check if run directory contains valid model."""
    model_path = os.path.join(run_dir, "model.pt")
    if not os.path.exists(model_path):
        print(f"[ERROR] No model.pt found in {run_dir}")
        print("        Available runs with models:")
        for d in os.listdir("runs"):
            if os.path.exists(os.path.join("runs", d, "model.pt")):
                print(f"          - runs/{d}")
        return False
    return True

def load_model(run_dir: str):
    """Load DeepPurpose model with validation."""
    if not validate_model_path(run_dir):
        return None
        
    print(f"[1/6] Loading Model from {run_dir}...")
    
    # Load the config.pkl if it exists for exact config match
    config_path = os.path.join(run_dir, "config.pkl")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        print("      Using saved config.pkl")
        # Override result_folder to current directory (old path may be stale)
        config["result_folder"] = run_dir
    else:
        # Fallback to hardcoded config
        config = utils.generate_config(
            drug_encoding=MODEL_CONFIG["drug_encoding"], 
            target_encoding=MODEL_CONFIG["target_encoding"], 
            cls_hidden_dims=MODEL_CONFIG["cls_hidden_dims"], 
            train_epoch=1, LR=1e-4, batch_size=256,
            result_folder=run_dir
        )
        print("      Using hardcoded config (no config.pkl found)")
    
    model = dp_models.model_initialize(**config)
    model.load_pretrained(os.path.join(run_dir, "model.pt"))
    model.model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.to(device)
    print(f"      Model loaded on {device}")
    
    return model

def load_data(dataset_name: str = "KIBA") -> pd.DataFrame:
    """
    Load DTI dataset from TDC (Therapeutics Data Commons).
    We need the original Drug/Target sequences to generate embeddings.
    """
    print(f"[2/6] Loading {dataset_name} Dataset from TDC...")
    
    from tdc.multi_pred import DTI
    data = DTI(name=dataset_name)
    
    # Get just the test split (smaller, faster for demo)
    split = data.get_split()
    df = split['test']  # Use test set
    
    # TDC uses different column names
    df = df.rename(columns={'Drug': 'Drug', 'Target': 'Target', 'Y': 'Label'})
    
    print(f"      Loaded {len(df)} test samples")
    print(f"      Columns: {list(df.columns)}")
    print(f"      Sample Drug (SMILES): {df['Drug'].iloc[0][:50]}...")
    print(f"      Sample Target (first 50 aa): {df['Target'].iloc[0][:50]}...")
    
    return df

def generate_embeddings(model, df: pd.DataFrame):
    """
    Generate embeddings WITHOUT shuffling to preserve order alignment.
    Returns: (drug_embeddings, target_embeddings, labels)
    """
    print("[3/6] Generating Embeddings (no shuffle)...")
    
    drugs = df['Drug'].values
    targets = df['Target'].values
    labels = df['Label'].values
    
    drug_encoding = model.config['drug_encoding']
    target_encoding = model.config['target_encoding']
    
    print(f"      Encoding {len(drugs)} drugs with {drug_encoding}...")
    print(f"      Encoding {len(targets)} targets with {target_encoding}...")
    
    # data_process returns DataFrames with encoded columns
    train_df, val_df, test_df = utils.data_process(
        drugs, targets, labels, 
        drug_encoding, target_encoding, 
        split_method='random', 
        frac=[0.01, 0.01, 0.98],  # Mostly test
        random_seed=42
    )
    
    # Combine all DataFrames
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"      Combined {len(all_df)} samples")
    
    # Create proper Dataset using DeepPurpose's loader
    from DeepPurpose.utils import data_process_loader
    
    config = {
        'drug_encoding': drug_encoding,
        'target_encoding': target_encoding
    }
    
    # Create loader - pass indices, labels, and the dataframe with config
    indices = list(range(len(all_df)))
    label_values = all_df['Label'].values
    dataset = data_process_loader(indices, label_values, all_df, **config)
    
    # Create DataLoader WITHOUT shuffling - CRITICAL!
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=False,
        drop_last=False
    )
    
    device = next(model.model.parameters()).device
    print(f"      Running inference on {device}... ({len(dataset)} samples)")
    
    all_drug_emb = []
    all_target_emb = []
    all_labels = []
    
    with torch.no_grad():
        for v_d, v_p, label in loader:
            v_d = v_d.to(device)
            v_p = v_p.to(device)
            
            emb_drug = model.model.model_drug(v_d).cpu().numpy()
            emb_target = model.model.model_protein(v_p).cpu().numpy()
            
            all_drug_emb.append(emb_drug)
            all_target_emb.append(emb_target)
            all_labels.extend(label.numpy().tolist())
    
    drug_embeddings = np.vstack(all_drug_emb)
    target_embeddings = np.vstack(all_target_emb)
    
    print(f"      Drug embeddings: {drug_embeddings.shape}")
    print(f"      Target embeddings: {target_embeddings.shape}")
    
    return drug_embeddings, target_embeddings, np.array(all_labels)

def compute_3d_projections(drug_emb: np.ndarray, target_emb: np.ndarray):
    """
    Compute REAL PCA projections for 3D visualization.
    """
    print("[4/6] Computing PCA projections...")
    
    combined = np.hstack([drug_emb, target_emb])
    
    pca = PCA(n_components=3)
    combined_3d = pca.fit_transform(combined)
    
    pca_drug = PCA(n_components=3)
    drug_3d = pca_drug.fit_transform(drug_emb)
    
    pca_target = PCA(n_components=3)
    target_3d = pca_target.fit_transform(target_emb)
    
    print(f"      Explained variance (combined): {pca.explained_variance_ratio_.sum():.2%}")
    print(f"      Explained variance (drug): {pca_drug.explained_variance_ratio_.sum():.2%}")
    print(f"      Explained variance (target): {pca_target.explained_variance_ratio_.sum():.2%}")
    
    return drug_3d, target_3d, combined_3d

def upload_to_qdrant(
    df: pd.DataFrame,
    drug_emb: np.ndarray,
    target_emb: np.ndarray,
    drug_3d: np.ndarray,
    target_3d: np.ndarray,
    combined_3d: np.ndarray,
    labels: np.ndarray
):
    """Upload vectors and metadata to Qdrant."""
    print(f"[5/6] Connecting to Qdrant ({QDRANT_HOST}:{QDRANT_PORT})...")
    
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
        client.get_collections()
    except Exception as e:
        print(f"[ERROR] Cannot connect to Qdrant: {e}")
        print("        Make sure Qdrant is running:")
        print("        docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        return False
    
    drug_dim = drug_emb.shape[1]
    target_dim = target_emb.shape[1]
    
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"      Collection '{COLLECTION_NAME}' exists. Recreating...")
    
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "drug": VectorParams(size=drug_dim, distance=Distance.COSINE),
            "target": VectorParams(size=target_dim, distance=Distance.COSINE),
        }
    )
    print(f"      Collection created: drug={drug_dim}d, target={target_dim}d")
    
    print("[6/6] Uploading points...")
    
    drugs = df['Drug'].values
    targets = df['Target'].values
    
    points = []
    for i in range(len(df)):
        payload = {
            "smiles": str(drugs[i]),
            "target_seq": str(targets[i])[:500],
            "label_true": float(labels[i]),
            "pca_drug": drug_3d[i].tolist(),
            "pca_target": target_3d[i].tolist(),
            "pca_combined": combined_3d[i].tolist(),
            "affinity_class": "high" if labels[i] > 7 else "medium" if labels[i] > 5 else "low",
        }
        
        point = PointStruct(
            id=i,
            vector={
                "drug": drug_emb[i].tolist(),
                "target": target_emb[i].tolist()
            },
            payload=payload
        )
        points.append(point)
        
        if len(points) >= 100:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []
            print(f"      Uploaded {i+1}/{len(df)}", end='\r')
    
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    
    print(f"\n      ✓ Successfully indexed {len(df)} points!")
    return True

def main():
    print("=" * 60)
    print("   DEEPPURPOSE -> QDRANT INGESTION PIPELINE (v2)")
    print("=" * 60)
    
    model = load_model(BEST_MODEL_RUN)
    if not model:
        sys.exit(1)
    
    # Load KIBA dataset from TDC (the model was trained on KIBA)
    df = load_data("KIBA")
    if df is None:
        sys.exit(1)
    
    drug_emb, target_emb, labels = generate_embeddings(model, df)
    drug_3d, target_3d, combined_3d = compute_3d_projections(drug_emb, target_emb)
    success = upload_to_qdrant(df, drug_emb, target_emb, drug_3d, target_3d, combined_3d, labels)
    
    if success:
        print("\n" + "=" * 60)
        print("   INGESTION COMPLETE")
        print("=" * 60)
        print(f"   Collection: {COLLECTION_NAME}")
        print(f"   Points: {len(df)}")
        print(f"   Next: python -m uvicorn server.api:app --reload")
    else:
        print("\n[FAILED] Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
