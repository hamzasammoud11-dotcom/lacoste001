"""
Base Ingestor - Abstract base class for data ingestion
========================================================

Provides common functionality for all ingestors:
- Rate limiting
- Batch processing
- Progress tracking
- Error handling
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime

from bioflow.core.base import Modality

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""
    source: str
    total_fetched: int
    total_indexed: int
    failed: int
    duration_seconds: float
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if self.total_fetched == 0:
            return 0.0
        return self.total_indexed / self.total_fetched
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "total_fetched": self.total_fetched,
            "total_indexed": self.total_indexed,
            "failed": self.failed,
            "success_rate": f"{self.success_rate:.2%}",
            "duration_seconds": round(self.duration_seconds, 2),
            "errors": self.errors[:10],  # Limit errors shown
        }


@dataclass
class DataRecord:
    """A single record to be ingested."""
    id: str
    content: str
    modality: str  # "text", "molecule", "protein"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseIngestor(ABC):
    """
    Abstract base class for data ingestors.
    
    Subclasses must implement:
    - fetch_data(): Retrieve data from source
    - parse_record(): Convert raw data to DataRecord
    """
    
    def __init__(
        self,
        qdrant_service,
        obm_encoder,
        collection: str = "bioflow_memory",
        batch_size: int = 50,
        rate_limit: float = 0.5,  # seconds between API calls
    ):
        """
        Initialize base ingestor.
        
        Args:
            qdrant_service: QdrantService instance for indexing
            obm_encoder: OBMEncoder instance for embeddings
            collection: Target Qdrant collection
            batch_size: Number of records per batch
            rate_limit: Minimum seconds between API calls
        """
        self.qdrant = qdrant_service
        self.encoder = obm_encoder
        self.collection = collection
        self.batch_size = batch_size
        self.rate_limit = rate_limit
        
        self._last_request_time = 0.0
        self._stats = {
            "fetched": 0,
            "indexed": 0,
            "failed": 0,
            "errors": [],
        }
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of the data source."""
        pass
    
    @abstractmethod
    def fetch_data(self, query: str, limit: int) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch data from the source.
        
        Args:
            query: Search query or identifier
            limit: Maximum records to fetch
            
        Yields:
            Raw data records from the source
        """
        pass
    
    @abstractmethod
    def parse_record(self, raw_data: Dict[str, Any]) -> Optional[DataRecord]:
        """
        Parse raw data into a DataRecord.
        
        Args:
            raw_data: Raw record from the source
            
        Returns:
            DataRecord or None if parsing fails
        """
        pass
    
    def _rate_limit_wait(self):
        """Wait to respect rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()
    
    def _get_modality_enum(self, modality_str: str) -> Optional[Modality]:
        """Convert string modality to Modality enum."""
        mapping = {
            "text": Modality.TEXT,
            "molecule": Modality.SMILES,
            "protein": Modality.PROTEIN,
            "image": Modality.IMAGE,
            "experiment": Modality.TEXT,  # Experiments use text encoding
        }
        return mapping.get(modality_str)
    
    def _encode_record(self, record: DataRecord) -> Optional[List[float]]:
        """Encode a record into a vector using OBMEncoder."""
        try:
            modality = self._get_modality_enum(record.modality)
            if modality is None:
                logger.warning(f"Unknown modality: {record.modality}")
                return None
            
            # Use unified encode() API
            result = self.encoder.encode(record.content, modality)
            
            if result and hasattr(result, 'vector'):
                # Vector may be list or numpy array
                vector = result.vector
                if hasattr(vector, 'tolist'):
                    return vector.tolist()
                return vector  # Already a list
            return None
            
        except Exception as e:
            logger.error(f"Encoding failed for {record.id}: {e}")
            return None
    
    def _index_batch(self, records: List[DataRecord]) -> int:
        """Index a batch of records into Qdrant."""
        indexed = 0
        
        for record in records:
            try:
                # Prepare metadata payload (store source_id in metadata, let Qdrant generate UUID)
                payload = {
                    "source": self.source_name,
                    "source_id": record.id,
                    "indexed_at": datetime.utcnow().isoformat(),
                    **record.metadata,
                }
                
                # Index using qdrant service (it handles encoding internally)
                # Don't pass id= since Qdrant requires UUIDs
                self.qdrant.ingest(
                    content=record.content,
                    modality=record.modality,
                    metadata=payload,
                    collection=self.collection,
                )
                
                indexed += 1
                
            except Exception as e:
                logger.error(f"Failed to index {record.id}: {e}")
                self._stats["errors"].append(f"{record.id}: {str(e)[:100]}")
                self._stats["failed"] += 1
        
        return indexed
    
    def ingest(self, query: str, limit: int = 100) -> IngestionResult:
        """
        Run the full ingestion pipeline.
        
        Args:
            query: Search query or topic
            limit: Maximum records to ingest
            
        Returns:
            IngestionResult with statistics
        """
        start_time = time.time()
        
        # Reset stats
        self._stats = {"fetched": 0, "indexed": 0, "failed": 0, "errors": []}
        
        logger.info(f"[{self.source_name}] Starting ingestion: query='{query}', limit={limit}")
        
        batch = []
        
        try:
            for raw_data in self.fetch_data(query, limit):
                self._stats["fetched"] += 1
                
                record = self.parse_record(raw_data)
                if record is None:
                    self._stats["failed"] += 1
                    continue
                
                batch.append(record)
                
                # Process batch when full
                if len(batch) >= self.batch_size:
                    indexed = self._index_batch(batch)
                    self._stats["indexed"] += indexed
                    batch = []
                    logger.info(f"[{self.source_name}] Progress: {self._stats['indexed']}/{self._stats['fetched']}")
            
            # Process remaining batch
            if batch:
                indexed = self._index_batch(batch)
                self._stats["indexed"] += indexed
                
        except Exception as e:
            logger.error(f"[{self.source_name}] Ingestion failed: {e}")
            self._stats["errors"].append(str(e))
        
        duration = time.time() - start_time
        
        result = IngestionResult(
            source=self.source_name,
            total_fetched=self._stats["fetched"],
            total_indexed=self._stats["indexed"],
            failed=self._stats["failed"],
            duration_seconds=duration,
            errors=self._stats["errors"],
        )
        
        logger.info(f"[{self.source_name}] Ingestion complete: {result.to_dict()}")
        
        return result
