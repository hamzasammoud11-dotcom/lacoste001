"""
Download Biomedical Images for BioFlow Demo
============================================

Downloads sample cell microscopy and biomedical images from public sources
for testing the image search functionality.
"""

import os
import json
import requests
from pathlib import Path
import time

# Target directory
DATA_DIR = Path(__file__).parent / "data" / "images"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Sample biomedical images from public sources
# Using PubChem compound structure images and public domain microscopy
BIOMEDICAL_IMAGES = [
    # PubChem compound structure images (for drug-related searches)
    {
        "url": "https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid=2244&t=l",
        "filename": "aspirin_structure.png",
        "description": "Aspirin (acetylsalicylic acid) 2D chemical structure",
        "image_type": "spectra",
        "modality": "molecule",
        "metadata": {
            "compound_id": "CHEMBL25",
            "compound_name": "Aspirin",
            "pubchem_cid": "2244",
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "source": "PubChem"
        }
    },
    {
        "url": "https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid=3672&t=l",
        "filename": "ibuprofen_structure.png",
        "description": "Ibuprofen 2D chemical structure",
        "image_type": "spectra",
        "modality": "molecule",
        "metadata": {
            "compound_id": "CHEMBL521",
            "compound_name": "Ibuprofen",
            "pubchem_cid": "3672",
            "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "source": "PubChem"
        }
    },
    {
        "url": "https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid=5090&t=l",
        "filename": "metformin_structure.png",
        "description": "Metformin 2D chemical structure - diabetes drug",
        "image_type": "spectra",
        "modality": "molecule",
        "metadata": {
            "compound_id": "CHEMBL1431",
            "compound_name": "Metformin",
            "pubchem_cid": "5090",
            "smiles": "CN(C)C(=N)N=C(N)N",
            "source": "PubChem"
        }
    },
    {
        "url": "https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid=36314&t=l",
        "filename": "atorvastatin_structure.png",
        "description": "Atorvastatin 2D chemical structure - cholesterol drug (Lipitor)",
        "image_type": "spectra",
        "modality": "molecule",
        "metadata": {
            "compound_id": "CHEMBL1487",
            "compound_name": "Atorvastatin",
            "pubchem_cid": "36314",
            "smiles": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
            "source": "PubChem"
        }
    },
    {
        "url": "https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid=5743&t=l",
        "filename": "omeprazole_structure.png",
        "description": "Omeprazole 2D chemical structure - proton pump inhibitor",
        "image_type": "spectra",
        "modality": "molecule",
        "metadata": {
            "compound_id": "CHEMBL1503",
            "compound_name": "Omeprazole",
            "pubchem_cid": "5743",
            "smiles": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC",
            "source": "PubChem"
        }
    },
    {
        "url": "https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid=60823&t=l",
        "filename": "sildenafil_structure.png",
        "description": "Sildenafil 2D chemical structure - PDE5 inhibitor (Viagra)",
        "image_type": "spectra",
        "modality": "molecule",
        "metadata": {
            "compound_id": "CHEMBL192",
            "compound_name": "Sildenafil",
            "pubchem_cid": "60823",
            "smiles": "CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
            "source": "PubChem"
        }
    },
    {
        "url": "https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid=5329102&t=l",
        "filename": "sorafenib_structure.png",
        "description": "Sorafenib 2D structure - kinase inhibitor cancer drug",
        "image_type": "spectra",
        "modality": "molecule",
        "metadata": {
            "compound_id": "CHEMBL1336",
            "compound_name": "Sorafenib",
            "pubchem_cid": "5329102",
            "smiles": "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F",
            "source": "PubChem",
            "target": "VEGFR, RAF kinase"
        }
    },
    {
        "url": "https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid=176870&t=l",
        "filename": "gefitinib_structure.png",
        "description": "Gefitinib 2D structure - EGFR tyrosine kinase inhibitor",
        "image_type": "spectra",
        "modality": "molecule",
        "metadata": {
            "compound_id": "CHEMBL939",
            "compound_name": "Gefitinib",
            "pubchem_cid": "176870",
            "smiles": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
            "source": "PubChem",
            "target": "EGFR"
        }
    },
    {
        "url": "https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid=2733526&t=l",
        "filename": "imatinib_structure.png",
        "description": "Imatinib 2D structure - BCR-ABL tyrosine kinase inhibitor (Gleevec)",
        "image_type": "spectra",
        "modality": "molecule",
        "metadata": {
            "compound_id": "CHEMBL941",
            "compound_name": "Imatinib",
            "pubchem_cid": "2733526",
            "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
            "source": "PubChem",
            "target": "BCR-ABL kinase"
        }
    },
    {
        "url": "https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid=5288826&t=l",
        "filename": "amlodipine_structure.png",
        "description": "Amlodipine 2D structure - calcium channel blocker",
        "image_type": "spectra",
        "modality": "molecule",
        "metadata": {
            "compound_id": "CHEMBL1491",
            "compound_name": "Amlodipine",
            "pubchem_cid": "5288826",
            "smiles": "CCOC(=O)C1=C(NC(=C(C1C2=CC=CC=C2Cl)C(=O)OC)C)COCCN",
            "source": "PubChem",
            "target": "L-type calcium channel"
        }
    },
    # Protein-related images (UniProt feature viewers as static images)
    {
        "url": "https://www.ebi.ac.uk/pdbe/static/entry/1bna_deposited_chain_front_image-200x200.png",
        "filename": "dna_helix_1bna.png",
        "description": "B-DNA crystal structure (PDB: 1BNA)",
        "image_type": "xray",
        "modality": "protein",
        "metadata": {
            "pdb_id": "1BNA",
            "type": "DNA helix",
            "resolution": "1.9 Angstrom",
            "source": "PDB/EMBL-EBI"
        }
    },
    {
        "url": "https://www.ebi.ac.uk/pdbe/static/entry/4hhb_deposited_chain_front_image-200x200.png",
        "filename": "hemoglobin_4hhb.png",
        "description": "Human hemoglobin structure (PDB: 4HHB)",
        "image_type": "xray",
        "modality": "protein",
        "metadata": {
            "pdb_id": "4HHB",
            "protein_name": "Hemoglobin",
            "uniprot_id": "P69905",
            "organism": "Homo sapiens",
            "source": "PDB/EMBL-EBI"
        }
    },
    {
        "url": "https://www.ebi.ac.uk/pdbe/static/entry/1hho_deposited_chain_front_image-200x200.png",
        "filename": "hemoglobin_oxygenated.png",
        "description": "Oxygenated hemoglobin structure",
        "image_type": "xray",
        "modality": "protein",
        "metadata": {
            "pdb_id": "1HHO",
            "protein_name": "Oxygenated Hemoglobin",
            "function": "Oxygen transport",
            "source": "PDB/EMBL-EBI"
        }
    },
    {
        "url": "https://www.ebi.ac.uk/pdbe/static/entry/1aoi_deposited_chain_front_image-200x200.png",
        "filename": "nucleosome_1aoi.png",
        "description": "Nucleosome core particle structure (PDB: 1AOI)",
        "image_type": "xray",
        "modality": "protein",
        "metadata": {
            "pdb_id": "1AOI",
            "complex_name": "Nucleosome",
            "function": "DNA packaging",
            "source": "PDB/EMBL-EBI"
        }
    },
    {
        "url": "https://www.ebi.ac.uk/pdbe/static/entry/2src_deposited_chain_front_image-200x200.png",
        "filename": "src_kinase_2src.png",
        "description": "SRC tyrosine kinase structure (PDB: 2SRC)",
        "image_type": "xray",
        "modality": "protein",
        "metadata": {
            "pdb_id": "2SRC",
            "protein_name": "SRC kinase",
            "uniprot_id": "P12931",
            "function": "Signal transduction, cancer target",
            "source": "PDB/EMBL-EBI"
        }
    },
    {
        "url": "https://www.ebi.ac.uk/pdbe/static/entry/1opk_deposited_chain_front_image-200x200.png",
        "filename": "egfr_kinase.png",
        "description": "EGFR tyrosine kinase domain with gefitinib",
        "image_type": "xray",
        "modality": "protein",
        "metadata": {
            "pdb_id": "1OPK",
            "protein_name": "EGFR",
            "uniprot_id": "P00533",
            "function": "Cancer drug target",
            "source": "PDB/EMBL-EBI"
        }
    },
    {
        "url": "https://www.ebi.ac.uk/pdbe/static/entry/3eml_deposited_chain_front_image-200x200.png",
        "filename": "abl_kinase_imatinib.png",
        "description": "ABL kinase in complex with imatinib (Gleevec)",
        "image_type": "xray",
        "modality": "protein",
        "metadata": {
            "pdb_id": "3EML",
            "protein_name": "ABL kinase",
            "ligand": "Imatinib",
            "function": "Cancer drug target",
            "source": "PDB/EMBL-EBI"
        }
    },
    {
        "url": "https://www.ebi.ac.uk/pdbe/static/entry/1m17_deposited_chain_front_image-200x200.png",
        "filename": "egfr_erlotinib.png",
        "description": "EGFR kinase with erlotinib (Tarceva)",
        "image_type": "xray",
        "modality": "protein",
        "metadata": {
            "pdb_id": "1M17",
            "protein_name": "EGFR",
            "ligand": "Erlotinib",
            "function": "NSCLC target",
            "source": "PDB/EMBL-EBI"
        }
    },
]

def download_image(image_info: dict) -> bool:
    """Download a single image."""
    filepath = DATA_DIR / image_info["filename"]
    
    # Skip if already exists
    if filepath.exists():
        print(f"  ✓ Already exists: {image_info['filename']}")
        return True
    
    try:
        headers = {
            "User-Agent": "BioFlow/1.0 (educational; contact: bioflow@example.com)"
        }
        response = requests.get(image_info["url"], headers=headers, timeout=30)
        response.raise_for_status()
        
        # Save image
        with open(filepath, "wb") as f:
            f.write(response.content)
        
        print(f"  ✓ Downloaded: {image_info['filename']} ({len(response.content)} bytes)")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed: {image_info['filename']} - {e}")
        return False


def create_manifest():
    """Create a JSON manifest file for ingestion."""
    manifest = []
    
    for img in BIOMEDICAL_IMAGES:
        filepath = DATA_DIR / img["filename"]
        if filepath.exists():
            manifest.append({
                "image": str(filepath.absolute()),
                "image_type": img["image_type"],
                "description": img["description"],
                "modality": img.get("modality", "other"),
                "source": "biomedical_sample_dataset",
                "metadata": img.get("metadata", {})
            })
    
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Created manifest: {manifest_path}")
    print(f"  Total images in manifest: {len(manifest)}")
    return manifest_path


def main():
    print("=" * 60)
    print("BioFlow Biomedical Image Dataset Downloader")
    print("=" * 60)
    print(f"\nTarget directory: {DATA_DIR}")
    print(f"Images to download: {len(BIOMEDICAL_IMAGES)}")
    print()
    
    successful = 0
    failed = 0
    
    for i, img in enumerate(BIOMEDICAL_IMAGES, 1):
        print(f"[{i}/{len(BIOMEDICAL_IMAGES)}] {img['description'][:50]}...")
        if download_image(img):
            successful += 1
        else:
            failed += 1
        time.sleep(0.5)  # Be nice to servers
    
    print(f"\n{'=' * 60}")
    print(f"Download complete: {successful} successful, {failed} failed")
    
    # Create manifest
    manifest_path = create_manifest()
    
    print(f"\n{'=' * 60}")
    print("Next steps:")
    print(f"1. Ingest images: Use the manifest at {manifest_path}")
    print("2. Or ingest directory: Use data/images as the source")
    print("=" * 60)


if __name__ == "__main__":
    main()
