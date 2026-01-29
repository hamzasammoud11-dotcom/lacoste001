"""
Image Ingestor - Biological Image Ingestion
============================================

Ingest biological images (microscopy, gels, spectra) into Qdrant.

Supports:
- Local file uploads
- URLs (with download)
- Base64 encoded images
- Batch ingestion from metadata manifest
"""

import logging
import os
from typing import List, Dict, Any, Optional, Generator, Union
from pathlib import Path
import json
from datetime import datetime
from io import BytesIO

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from bioflow.ingestion.base_ingestor import BaseIngestor, IngestionResult, DataRecord
from bioflow.core.base import Modality

logger = logging.getLogger(__name__)


class ImageIngestor(BaseIngestor):
    """
    Ingestor for biological images.
    
    Supports multiple image types:
    - microscopy: Cell/tissue microscopy
    - gel: Gel electrophoresis
    - spectra: Spectroscopy data (NMR, MS, etc.)
    - xray: X-ray crystallography
    - other: Other biological images
    """
    
    @property
    def source_name(self) -> str:
        return "image"
    
    def fetch_data(
        self,
        query: str,
        limit: int
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch images from query.
        
        For images, 'query' can be:
        - Path to metadata JSON file
        - Directory path containing images
        - Not applicable for API uploads (use batch_ingest instead)
        """
        # If query is a JSON file, load metadata
        if query.endswith('.json') and os.path.exists(query):
            logger.info(f"Loading images from metadata file: {query}")
            with open(query, 'r') as f:
                metadata_records = json.load(f)
            
            for record in metadata_records[:limit]:
                yield record
        
        # If query is a directory, scan for images
        elif os.path.isdir(query):
            logger.info(f"Scanning directory for images: {query}")
            image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
            
            count = 0
            for root, _, files in os.walk(query):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        if count >= limit:
                            return
                        
                        filepath = os.path.join(root, file)
                        yield {
                            "image": filepath,
                            "image_type": "other",
                            "source": "local_scan",
                            "description": f"Image from {filepath}"
                        }
                        count += 1
        else:
            logger.warning(f"Query '{query}' is neither a JSON file nor a directory")
    
    def parse_record(self, raw_data: Dict[str, Any]) -> Optional[DataRecord]:
        """
        Parse image record.
        
        Expected raw_data format:
        {
            "image": PIL.Image | bytes | str,  # PIL Image (preferred), bytes, file path, or URL
            "image_type": str,                  # microscopy, gel, spectra, etc.
            "experiment_id": str,
            "source": str,
            "description": str,
            "metadata": dict
        }
        
        IN-MEMORY STREAMING: Pass PIL Images directly for zero disk usage (like text ingestion).
        """
        try:
            image_data = raw_data.get("image") or raw_data.get("local_path")
            if not image_data:
                logger.warning("No image data provided in record")
                return None
            
            # Handle different image formats
            content = image_data  # Default: pass as-is (path/URL for ImageEncoder)
            
            # If PIL Image (in-memory streaming - PREFERRED), use directly
            if PIL_AVAILABLE and isinstance(image_data, Image.Image):
                content = image_data
                # Generate UUID from image bytes for uniqueness
                import hashlib
                import uuid
                from io import BytesIO
                img_bytes = BytesIO()
                image_data.save(img_bytes, format='PNG')
                # Generate UUID v5 from image hash
                image_hash = hashlib.sha256(img_bytes.getvalue()).hexdigest()
                image_id = str(uuid.uuid5(uuid.NAMESPACE_OID, image_hash))
            
            # If bytes, convert to PIL Image
            elif isinstance(image_data, bytes):
                if PIL_AVAILABLE:
                    content = Image.open(BytesIO(image_data))
                import hashlib
                import uuid
                image_hash = hashlib.sha256(image_data).hexdigest()
                image_id = str(uuid.uuid5(uuid.NAMESPACE_OID, image_hash))
            
            # If string (path/URL), generate UUID from string
            else:
                import hashlib
                import uuid
                image_hash = hashlib.sha256(str(image_data).encode()).hexdigest()
                image_id = str(uuid.uuid5(uuid.NAMESPACE_OID, image_hash))
            
            # Build metadata
            metadata = {
                "source": raw_data.get("source", "upload"),
                "source_id": raw_data.get("source_id", f"image:{image_id}"),
                "image_type": raw_data.get("image_type", "other"),
                "experiment_id": raw_data.get("experiment_id", ""),
                "description": raw_data.get("description", ""),
                "caption": raw_data.get("caption", ""),
                "indexed_at": datetime.now().isoformat(),
                **raw_data.get("metadata", {})
            }
            
            # Store image data as content (will be encoded later)
            return DataRecord(
                id=image_id,  # Proper UUID
                content=image_data,  # Raw image data (path, bytes, or base64)
                modality="image",
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to parse image record: {e}")
            return None
    
    def ingest_from_metadata(
        self,
        metadata_file: str,
        collection: Optional[str] = None
    ) -> IngestionResult:
        """
        Ingest images from a metadata JSON file.
        
        This is the recommended method for batch ingestion.
        
        Args:
            metadata_file: Path to JSON file with image metadata
            collection: Target collection (default: self.collection)
        
        Returns:
            IngestionResult with statistics
        """
        collection = collection or self.collection
        start_time = datetime.now()
        
        logger.info(f"Starting ingestion from metadata file: {metadata_file}")
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata_records = json.load(f)
        
        logger.info(f"Found {len(metadata_records)} images in metadata")
        
        indexed = 0
        failed = 0
        errors = []
        
        for raw_record in metadata_records:
            try:
                # Parse record
                record = self.parse_record(raw_record)
                if not record:
                    failed += 1
                    continue
                
                # Encode image
                modality_enum = self._get_modality_enum(record.modality)
                if not modality_enum:
                    logger.error(f"Unsupported modality: {record.modality}")
                    failed += 1
                    continue
                
                self._rate_limit_wait()
                
                embedding_results = self.encoder.encode(
                    record.content,
                    modality=modality_enum
                )
                
                # Handle single result
                if not isinstance(embedding_results, list):
                    embedding_results = [embedding_results]
                
                if not embedding_results:
                    failed += 1
                    continue
                
                embedding_result = embedding_results[0]
                
                # Check for encoding errors
                if "error" in embedding_result.metadata:
                    logger.error(f"Encoding error: {embedding_result.metadata['error']}")
                    failed += 1
                    errors.append(embedding_result.metadata['error'])
                    continue
                
                # Store in Qdrant
                content_str = record.content if isinstance(record.content, str) else "[binary image]"
                
                from qdrant_client.models import PointStruct
                import numpy as np
                
                point = PointStruct(
                    id=record.id,
                    vector=np.array(embedding_result.vector).tolist(),
                    payload={
                        "content": content_str[:500],  # Truncate for storage
                        "modality": record.modality,
                        **record.metadata
                    }
                )
                
                self.qdrant._get_client().upsert(
                    collection_name=collection,
                    points=[point]
                )
                
                indexed += 1
                
                if indexed % 10 == 0:
                    logger.info(f"Progress: {indexed} images indexed, {failed} failed")
                
            except Exception as e:
                logger.error(f"Failed to ingest image: {e}")
                errors.append(str(e))
                failed += 1
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = IngestionResult(
            source=self.source_name,
            total_fetched=len(metadata_records),
            total_indexed=indexed,
            failed=failed,
            duration_seconds=duration,
            errors=errors[:10]  # Limit errors
        )
        
        logger.info(f"Γ£à Ingestion complete: {indexed} indexed, {failed} failed in {duration:.2f}s")
        
        return result
    
    def batch_ingest(
        self,
        images: List[Dict[str, Any]],
        collection: Optional[str] = None
    ) -> IngestionResult:
        """
        Batch ingest multiple images.
        
        Args:
            images: List of image records (see parse_record for format)
            collection: Target collection (default: self.collection)
        
        Returns:
            IngestionResult with statistics
        """
        collection = collection or self.collection
        start_time = datetime.now()
        
        logger.info(f"Starting batch ingestion of {len(images)} images...")
        
        indexed = 0
        failed = 0
        errors = []
        
        for img_data in images:
            try:
                # Parse record
                record = self.parse_record(img_data)
                if not record:
                    failed += 1
                    continue
                
                # Encode image
                modality_enum = self._get_modality_enum(record.modality)
                if not modality_enum:
                    logger.error(f"Unsupported modality: {record.modality}")
                    failed += 1
                    continue
                
                self._rate_limit_wait()
                
                embedding_results = self.encoder.encode(
                    record.content,
                    modality=modality_enum
                )
                
                # Handle single result
                if not isinstance(embedding_results, list):
                    embedding_results = [embedding_results]
                
                if not embedding_results:
                    failed += 1
                    continue
                
                embedding_result = embedding_results[0]
                
                # Check for encoding errors
                if "error" in embedding_result.metadata:
                    logger.error(f"Encoding error: {embedding_result.metadata['error']}")
                    failed += 1
                    continue
                
                # Store in Qdrant
                content_str = record.content if isinstance(record.content, str) else "[binary image]"
                
                from qdrant_client.models import PointStruct
                import numpy as np
                
                point = PointStruct(
                    id=record.id,
                    vector=np.array(embedding_result.vector).tolist(),
                    payload={
                        "content": content_str[:500],
                        "modality": record.modality,
                        **record.metadata
                    }
                )
                
                self.qdrant._get_client().upsert(
                    collection_name=collection,
                    points=[point]
                )
                
                indexed += 1
                
            except Exception as e:
                logger.error(f"Failed to ingest image: {e}")
                errors.append(str(e))
                failed += 1
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = IngestionResult(
            source=self.source_name,
            total_fetched=len(images),
            total_indexed=indexed,
            failed=failed,
            duration_seconds=duration,
            errors=errors
        )
        
        logger.info(f"Batch ingestion complete: {indexed} indexed, {failed} failed")
        
        return result
