#!/usr/bin/env python
"""
BioFlow Data Ingestion Script
==============================

Unified script to ingest data from multiple biological databases.

Usage:
    python -m bioflow.ingestion.ingest_all --query "EGFR lung cancer" --limit 100
    
    # Or programmatically:
    from bioflow.ingestion.ingest_all import run_full_ingestion
    results = run_full_ingestion("EGFR", pubmed_limit=100, uniprot_limit=50, chembl_limit=30)
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from bioflow.ingestion.base_ingestor import IngestionResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def get_services():
    """Initialize and return QdrantService and OBMEncoder."""
    from bioflow.api.model_service import get_model_service
    from bioflow.api.qdrant_service import get_qdrant_service
    
    logger.info("Initializing services...")
    
    model_service = get_model_service(lazy_load=True)
    qdrant_service = get_qdrant_service(model_service=model_service)
    
    # Get OBM encoder from model service
    obm_encoder = model_service.get_obm_encoder()
    
    return qdrant_service, obm_encoder


def run_pubmed_ingestion(
    qdrant_service,
    obm_encoder,
    query: str,
    limit: int = 100,
    collection: str = "bioflow_memory",
) -> IngestionResult:
    """Run PubMed ingestion."""
    from bioflow.ingestion.pubmed_ingestor import PubMedIngestor
    
    ingestor = PubMedIngestor(
        qdrant_service=qdrant_service,
        obm_encoder=obm_encoder,
        collection=collection,
    )
    
    return ingestor.ingest(query, limit)


def run_uniprot_ingestion(
    qdrant_service,
    obm_encoder,
    query: str,
    limit: int = 50,
    collection: str = "bioflow_memory",
) -> IngestionResult:
    """Run UniProt ingestion."""
    from bioflow.ingestion.uniprot_ingestor import UniProtIngestor
    
    ingestor = UniProtIngestor(
        qdrant_service=qdrant_service,
        obm_encoder=obm_encoder,
        collection=collection,
    )
    
    return ingestor.ingest(query, limit)


def run_chembl_ingestion(
    qdrant_service,
    obm_encoder,
    query: str,
    limit: int = 30,
    collection: str = "bioflow_memory",
    search_mode: str = "target",
) -> IngestionResult:
    """Run ChEMBL ingestion."""
    from bioflow.ingestion.chembl_ingestor import ChEMBLIngestor
    
    ingestor = ChEMBLIngestor(
        qdrant_service=qdrant_service,
        obm_encoder=obm_encoder,
        collection=collection,
        search_mode=search_mode,
    )
    
    return ingestor.ingest(query, limit)


def run_image_ingestion(
    qdrant_service,
    obm_encoder,
    query: str,
    limit: int = 1000,
    collection: str = "bioflow_memory",
    batch_size: int = 20,
) -> IngestionResult:
    """
    Run biomedical image ingestion from IDR, PubChem, and PMC.
    
    Images are streamed in-memory and ingested in batches to avoid RAM bloating.
    
    Args:
        qdrant_service: Qdrant service instance
        obm_encoder: OBM encoder instance
        query: Search query to guide image selection
        limit: Total images to ingest (default: 1000 to match ~1% of DAVIS size)
        collection: Target collection
        batch_size: Images per batch (default: 20 to avoid RAM issues)
    """
    from bioflow.ingestion.image_ingestor import ImageIngestor
    from bioflow.ingestion.image_sources import stream_biomedical_images
    import time
    
    start_time = time.time()
    
    ingestor = ImageIngestor(
        qdrant_service=qdrant_service,
        obm_encoder=obm_encoder,
        collection=collection,
    )
    
    logger.info(f"Starting image ingestion (limit={limit}, batch_size={batch_size})...")
    logger.info(f"Query: '{query}'")
    logger.info("Sources: IDR (microscopy), PubChem (spectra), PMC (gels)")
    
    total_indexed = 0
    total_failed = 0
    errors = []
    batch = []
    
    try:
        # Calculate max per type (equal distribution)
        max_per_type = limit // 3
        
        # Stream images in batches
        for i, image_record in enumerate(stream_biomedical_images(
            limit=limit,
            max_per_type=max_per_type,
            query=query
        ), 1):
            batch.append(image_record)
            
            # Process batch when full
            if len(batch) >= batch_size:
                try:
                    result = ingestor.batch_ingest(batch, collection=collection)
                    total_indexed += result.total_indexed
                    total_failed += result.failed
                    
                    logger.info(
                        f"Batch {i//batch_size}: Indexed {result.total_indexed}/{len(batch)} images "
                        f"(Total: {total_indexed}/{limit})"
                    )
                    
                    # Clear batch to free RAM
                    batch.clear()
                    
                except Exception as e:
                    logger.error(f"Batch ingestion failed: {e}")
                    errors.append(str(e))
                    total_failed += len(batch)
                    batch.clear()
        
        # Process remaining images
        if batch:
            try:
                result = ingestor.batch_ingest(batch, collection=collection)
                total_indexed += result.total_indexed
                total_failed += result.failed
                logger.info(f"Final batch: Indexed {result.total_indexed}/{len(batch)} images")
            except Exception as e:
                logger.error(f"Final batch failed: {e}")
                errors.append(str(e))
                total_failed += len(batch)
    
    except Exception as e:
        logger.error(f"Image streaming failed: {e}")
        errors.append(str(e))
    
    duration = time.time() - start_time
    
    return IngestionResult(
        source="images",
        total_fetched=total_indexed + total_failed,
        total_indexed=total_indexed,
        failed=total_failed,
        duration_seconds=duration,
        errors=errors
    )


def run_full_ingestion(
    query: str,
    pubmed_limit: int = 100,
    uniprot_limit: int = 50,
    chembl_limit: int = 30,
    image_limit: int = 1000,
    collection: str = "bioflow_memory",
    skip_pubmed: bool = False,
    skip_uniprot: bool = False,
    skip_chembl: bool = False,
    skip_images: bool = False,
) -> Dict[str, IngestionResult]:
    """
    Run full ingestion pipeline across all sources.
    
    Args:
        query: Search query (applied to text/molecule/protein sources)
        pubmed_limit: Max PubMed articles
        uniprot_limit: Max UniProt proteins
        chembl_limit: Max ChEMBL molecules
        image_limit: Max biomedical images (IDR, PubChem, PMC)
        collection: Target Qdrant collection
        skip_*: Skip specific sources
        
    Returns:
        Dictionary of source -> IngestionResult
        
    Note:
        Image limit of 1000 is ~1% of DAVIS (25K) or ~0.8% of KIBA (118K).
        Adjust based on your needs - images are streamed in batches to avoid RAM bloating.
    """
    results = {}
    
    # Initialize services
    qdrant_service, obm_encoder = get_services()
    
    logger.info("=" * 60)
    logger.info(f"BioFlow Full Ingestion - Query: '{query}'")
    logger.info("=" * 60)
    
    # PubMed
    if not skip_pubmed:
        logger.info("\n📚 Starting PubMed ingestion...")
        try:
            results["pubmed"] = run_pubmed_ingestion(
                qdrant_service, obm_encoder, query, pubmed_limit, collection
            )
        except Exception as e:
            logger.error(f"PubMed ingestion failed: {e}")
            results["pubmed"] = IngestionResult(
                source="pubmed", total_fetched=0, total_indexed=0,
                failed=0, duration_seconds=0, errors=[str(e)]
            )
    
    # UniProt
    if not skip_uniprot:
        logger.info("\n🧬 Starting UniProt ingestion...")
        try:
            # Adapt query for UniProt (add organism filter for human)
            uniprot_query = f"{query} AND organism_id:9606"
            results["uniprot"] = run_uniprot_ingestion(
                qdrant_service, obm_encoder, uniprot_query, uniprot_limit, collection
            )
        except Exception as e:
            logger.error(f"UniProt ingestion failed: {e}")
            results["uniprot"] = IngestionResult(
                source="uniprot", total_fetched=0, total_indexed=0,
                failed=0, duration_seconds=0, errors=[str(e)]
            )
    
    # ChEMBL
    if not skip_chembl:
        logger.info("\n💊 Starting ChEMBL ingestion...")
        try:
            results["chembl"] = run_chembl_ingestion(
                qdrant_service, obm_encoder, query, chembl_limit, collection
            )
        except Exception as e:
            logger.error(f"ChEMBL ingestion failed: {e}")
            results["chembl"] = IngestionResult(
                source="chembl", total_fetched=0, total_indexed=0,
                failed=0, duration_seconds=0, errors=[str(e)]
            )
    
    # Images
    if not skip_images:
        logger.info("\n🖼️  Starting Image ingestion...")
        try:
            results["images"] = run_image_ingestion(
                qdrant_service, obm_encoder, query, image_limit, collection
            )
        except Exception as e:
            logger.error(f"Image ingestion failed: {e}")
            results["images"] = IngestionResult(
                source="images", total_fetched=0, total_indexed=0,
                failed=0, duration_seconds=0, errors=[str(e)]
            )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)
    
    total_indexed = 0
    for source, result in results.items():
        logger.info(f"  {source.upper():10} | Indexed: {result.total_indexed:4} | "
                   f"Failed: {result.failed:4} | Time: {result.duration_seconds:.1f}s")
        total_indexed += result.total_indexed
    
    logger.info("-" * 60)
    logger.info(f"  {'TOTAL':10} | Indexed: {total_indexed:4}")
    logger.info("=" * 60)
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BioFlow Data Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest data about EGFR and lung cancer (including images)
  python -m bioflow.ingestion.ingest_all --query "EGFR lung cancer" --limit 100
  
  # Ingest only images (1000 images by default)
  python -m bioflow.ingestion.ingest_all --query "kinase" --skip-pubmed --skip-uniprot --skip-chembl
  
  # Large ingestion with custom limits for each source
  python -m bioflow.ingestion.ingest_all --query "kinase inhibitor" --pubmed-limit 500 --uniprot-limit 200 --chembl-limit 100 --image-limit 2000
  
  # Skip image ingestion (text/molecules/proteins only)
  python -m bioflow.ingestion.ingest_all --query "BRCA1" --skip-images
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Search query (applied to all sources)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=100,
        help="Default limit for all sources (default: 100)"
    )
    parser.add_argument(
        "--pubmed-limit",
        type=int,
        help="Override limit for PubMed"
    )
    parser.add_argument(
        "--uniprot-limit",
        type=int,
        help="Override limit for UniProt"
    )
    parser.add_argument(
        "--chembl-limit",
        type=int,
        help="Override limit for ChEMBL"
    )
    parser.add_argument(
        "--image-limit",
        type=int,
        default=1000,
        help="Override limit for images (default: 1000, ~1%% of DAVIS size)"
    )
    parser.add_argument(
        "--collection",
        default="bioflow_memory",
        help="Qdrant collection name (default: bioflow_memory)"
    )
    parser.add_argument(
        "--skip-pubmed",
        action="store_true",
        help="Skip PubMed ingestion"
    )
    parser.add_argument(
        "--skip-uniprot",
        action="store_true",
        help="Skip UniProt ingestion"
    )
    parser.add_argument(
        "--skip-chembl",
        action="store_true",
        help="Skip ChEMBL ingestion"
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image ingestion (IDR, PubChem, PMC)"
    )
    
    args = parser.parse_args()
    
    # Run ingestion
    results = run_full_ingestion(
        query=args.query,
        pubmed_limit=args.pubmed_limit or args.limit,
        uniprot_limit=args.uniprot_limit or (args.limit // 2),
        chembl_limit=args.chembl_limit or (args.limit // 3),
        image_limit=args.image_limit,
        collection=args.collection,
        skip_pubmed=args.skip_pubmed,
        skip_uniprot=args.skip_uniprot,
        skip_chembl=args.skip_chembl,
        skip_images=args.skip_images,
    )
    
    # Return success if any data was indexed
    total = sum(r.total_indexed for r in results.values())
    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
