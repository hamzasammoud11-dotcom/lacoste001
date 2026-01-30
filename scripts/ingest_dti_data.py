"""
Ingest KIBA/DAVIS Drug-Target Interaction datasets into Qdrant.

Uses OBMEncoder (768-dim) to create searchable vectors from real DTI data.
"""
import sys
import os
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import pandas as pd
from tqdm import tqdm

from bioflow.api.qdrant_service import get_qdrant_service


def load_dataset(dataset_name: str, limit: int = None) -> pd.DataFrame:
    """Load KIBA or DAVIS dataset from local .tab files."""
    filepath = os.path.join(ROOT_DIR, "data", f"{dataset_name.lower()}.tab")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    print(f"Loading {dataset_name} from {filepath}...")
    df = pd.read_csv(filepath, sep='\t')
    
    # Rename columns for consistency
    # Format: ID1, X1 (SMILES), ID2, X2 (sequence), Y (affinity)
    df.columns = ['drug_id', 'smiles', 'target_id', 'target_seq', 'affinity']
    
    # Remove duplicates (keep unique drug-target pairs)
    df = df.drop_duplicates(subset=['smiles', 'target_id'])
    
    if limit:
        df = df.head(limit)
    
    print(f"  Loaded {len(df)} unique drug-target pairs")
    return df


def get_affinity_class(affinity: float, dataset: str) -> str:
    """Classify affinity into high/medium/low based on dataset thresholds."""
    if dataset.upper() == "KIBA":
        # KIBA: lower is better (inhibition constant)
        if affinity < 6:
            return "high"
        elif affinity < 8:
            return "medium"
        else:
            return "low"
    else:  # DAVIS
        # DAVIS: Kd values, lower is better
        if affinity < 6:
            return "high"
        elif affinity < 7:
            return "medium"
        else:
            return "low"


def get_drug_name(drug_id, smiles: str) -> str:
    """Generate a readable drug name from ID or SMILES."""
    drug_id_str = str(drug_id)
    # If drug_id is numeric (like PubChem ID), create a friendly name
    if drug_id_str.isdigit():
        # Use PubChem CID format for known numeric IDs
        return f"CID-{drug_id_str}"
    return drug_id_str


def ingest_molecules(qdrant, df: pd.DataFrame, dataset: str, batch_size: int = 50):
    """Ingest unique molecules (drugs) from the dataset."""
    print("\n[1/2] Ingesting molecules (drugs)...")
    
    # Get unique SMILES with their best affinity
    unique_drugs = df.groupby('smiles').agg({
        'drug_id': 'first',
        'affinity': 'min',  # Best affinity
        'target_id': 'count'  # Number of targets
    }).reset_index()
    unique_drugs.columns = ['smiles', 'drug_id', 'best_affinity', 'num_targets']
    
    print(f"  Found {len(unique_drugs)} unique molecules")
    
    success_count = 0
    for idx, row in tqdm(unique_drugs.iterrows(), total=len(unique_drugs), desc="  Molecules"):
        try:
            affinity_class = get_affinity_class(row['best_affinity'], dataset)
            drug_name = get_drug_name(row['drug_id'], row['smiles'])
            
            result = qdrant.ingest(
                content=row['smiles'],
                modality="molecule",
                metadata={
                    "name": drug_name,
                    "drug_id": str(row['drug_id']),  # Keep original ID
                    "smiles": row['smiles'],
                    "description": f"Drug from {dataset.upper()} dataset",
                    "source": dataset.lower(),
                    "dataset": dataset.lower(),
                    "best_affinity": float(row['best_affinity']),
                    "affinity_class": affinity_class,
                    "num_targets": int(row['num_targets']),
                }
            )
            success_count += 1
        except Exception as e:
            if success_count == 0:
                print(f"\n  First error: {e}")  # Show first error for debugging
    
    print(f"  âœ“ Ingested {success_count}/{len(unique_drugs)} molecules")
    return success_count


def ingest_proteins(qdrant, df: pd.DataFrame, dataset: str, batch_size: int = 50):
    """Ingest unique proteins (targets) from the dataset."""
    print("\n[2/2] Ingesting proteins (targets)...")
    
    # Get unique proteins with their best affinity
    unique_targets = df.groupby('target_id').agg({
        'target_seq': 'first',
        'affinity': 'min',  # Best affinity
        'smiles': 'count'  # Number of drugs
    }).reset_index()
    unique_targets.columns = ['target_id', 'target_seq', 'best_affinity', 'num_drugs']
    
    print(f"  Found {len(unique_targets)} unique proteins")
    
    success_count = 0
    for idx, row in tqdm(unique_targets.iterrows(), total=len(unique_targets), desc="  Proteins"):
        try:
            # Truncate very long sequences for embedding
            sequence = str(row['target_seq'])[:1000]
            affinity_class = get_affinity_class(row['best_affinity'], dataset)
            
            result = qdrant.ingest(
                content=sequence,
                modality="protein",
                metadata={
                    "name": row['target_id'],
                    "uniprot_id": row['target_id'],
                    "sequence": sequence,
                    "full_length": len(str(row['target_seq'])),
                    "description": f"Target from {dataset.upper()} dataset",
                    "source": dataset.lower(),
                    "dataset": dataset.lower(),
                    "best_affinity": float(row['best_affinity']),
                    "affinity_class": affinity_class,
                    "num_drugs": int(row['num_drugs']),
                }
            )
            success_count += 1
        except Exception as e:
            if success_count == 0:
                print(f"\n  First error: {e}")  # Show first error for debugging
    
    print(f"  âœ“ Ingested {success_count}/{len(unique_targets)} proteins")
    return success_count


def main():
    parser = argparse.ArgumentParser(description="Ingest KIBA/DAVIS datasets into Qdrant")
    parser.add_argument("--dataset", choices=["kiba", "davis", "both"], default="davis",
                        help="Dataset to ingest (default: davis)")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Limit number of records per dataset (default: 1000, 0 for all)")
    parser.add_argument("--clear", action="store_true",
                        help="Clear existing collections before ingesting")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  KIBA/DAVIS -> QDRANT INGESTION")
    print("=" * 60)
    
    qdrant = get_qdrant_service()
    
    if args.clear:
        print("\nClearing existing collections...")
        try:
            client = qdrant._get_client()
            for coll in qdrant.list_collections():
                client.delete_collection(coll)
                print(f"  Deleted: {coll}")
            # Clear the cache so collections will be recreated
            qdrant._initialized_collections.clear()
        except Exception as e:
            print(f"  Warning: {e}")
    
    datasets = ["kiba", "davis"] if args.dataset == "both" else [args.dataset]
    limit = args.limit if args.limit > 0 else None
    
    total_molecules = 0
    total_proteins = 0
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"  Processing {dataset.upper()}")
        print("=" * 60)
        
        try:
            df = load_dataset(dataset, limit=limit)
            total_molecules += ingest_molecules(qdrant, df, dataset)
            total_proteins += ingest_proteins(qdrant, df, dataset)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Total molecules: {total_molecules}")
    print(f"  Total proteins: {total_proteins}")
    print(f"\nSearch at: http://localhost:3000/dashboard/discovery")


if __name__ == "__main__":
    main()

