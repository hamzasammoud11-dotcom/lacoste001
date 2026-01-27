"""
UniProt Ingestor - Protein Sequence Ingestion
===============================================

Fetches protein sequences from UniProt using their REST API.
https://www.uniprot.org/help/api

Usage:
    from bioflow.ingestion import UniProtIngestor
    
    ingestor = UniProtIngestor(qdrant_service, obm_encoder)
    result = ingestor.ingest("EGFR human", limit=50)
"""

import logging
import requests
from typing import Dict, Any, Optional, Generator

from bioflow.ingestion.base_ingestor import BaseIngestor, DataRecord

logger = logging.getLogger(__name__)


class UniProtIngestor(BaseIngestor):
    """
    Ingestor for UniProt protein sequences.
    
    Uses UniProt REST API to search and retrieve protein data.
    """
    
    SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
    
    def __init__(
        self,
        qdrant_service,
        obm_encoder,
        collection: str = "bioflow_memory",
        batch_size: int = 50,
        rate_limit: float = 0.2,  # UniProt is generous with rate limits
    ):
        """Initialize UniProt ingestor."""
        super().__init__(qdrant_service, obm_encoder, collection, batch_size, rate_limit)
    
    @property
    def source_name(self) -> str:
        return "uniprot"
    
    def fetch_data(self, query: str, limit: int) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch proteins matching query from UniProt.
        
        Args:
            query: UniProt search query (e.g., "EGFR AND organism_id:9606")
            limit: Maximum proteins to fetch
            
        Yields:
            Protein data dictionaries
        """
        # Prepare query - add reviewed filter for quality
        full_query = f"({query})"
        
        params = {
            "query": full_query,
            "format": "json",
            "size": min(limit, 500),  # UniProt page size limit
            "fields": "accession,id,protein_name,gene_names,organism_name,organism_id,length,sequence,cc_function,ft_domain,xref_pdb",
        }
        
        fetched = 0
        next_link = None
        
        while fetched < limit:
            try:
                self._rate_limit_wait()
                
                if next_link:
                    response = requests.get(next_link, timeout=30)
                else:
                    response = requests.get(self.SEARCH_URL, params=params, timeout=30)
                
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                
                if not results:
                    break
                
                for protein in results:
                    if fetched >= limit:
                        break
                    yield protein
                    fetched += 1
                
                # Check for pagination
                next_link = None
                link_header = response.headers.get("Link", "")
                if 'rel="next"' in link_header:
                    # Parse next link from header
                    for part in link_header.split(","):
                        if 'rel="next"' in part:
                            next_link = part.split(";")[0].strip("<> ")
                            break
                
                if not next_link:
                    break
                    
            except Exception as e:
                logger.error(f"[UniProt] Fetch failed: {e}")
                break
        
        logger.info(f"[UniProt] Fetched {fetched} proteins for query: {query}")
    
    def parse_record(self, raw_data: Dict[str, Any]) -> Optional[DataRecord]:
        """
        Parse a UniProt entry into a DataRecord.
        
        Args:
            raw_data: UniProt JSON entry
            
        Returns:
            DataRecord for protein embedding
        """
        try:
            accession = raw_data.get("primaryAccession", "")
            
            if not accession:
                return None
            
            # Get sequence
            sequence = raw_data.get("sequence", {}).get("value", "")
            
            if not sequence:
                logger.debug(f"[UniProt] Skipping {accession}: no sequence")
                return None
            
            # Extract protein name
            protein_name = ""
            protein_desc = raw_data.get("proteinDescription", {})
            rec_name = protein_desc.get("recommendedName", {})
            if rec_name:
                full_name = rec_name.get("fullName", {})
                protein_name = full_name.get("value", "")
            
            if not protein_name:
                # Try alternative names
                alt_names = protein_desc.get("alternativeNames", [])
                if alt_names:
                    protein_name = alt_names[0].get("fullName", {}).get("value", "")
            
            # Extract gene names
            gene_names = []
            for gene in raw_data.get("genes", []):
                if gene.get("geneName"):
                    gene_names.append(gene["geneName"].get("value", ""))
            
            # Extract organism
            organism = raw_data.get("organism", {}).get("scientificName", "")
            organism_id = raw_data.get("organism", {}).get("taxonId", "")
            
            # Extract function annotation
            function_text = ""
            comments = raw_data.get("comments", [])
            for comment in comments:
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        function_text = texts[0].get("value", "")
                        break
            
            # Extract PDB cross-references
            pdb_ids = []
            xrefs = raw_data.get("uniProtKBCrossReferences", [])
            for xref in xrefs:
                if xref.get("database") == "PDB":
                    pdb_ids.append(xref.get("id", ""))
            
            # Sequence length
            seq_length = raw_data.get("sequence", {}).get("length", len(sequence))
            
            return DataRecord(
                id=f"uniprot:{accession}",
                content=sequence,
                modality="protein",
                metadata={
                    "accession": accession,
                    "entry_name": raw_data.get("uniProtkbId", ""),
                    "protein_name": protein_name,
                    "gene_names": gene_names,
                    "organism": organism,
                    "organism_id": organism_id,
                    "function": function_text[:500],  # Truncate
                    "sequence_length": seq_length,
                    "pdb_ids": pdb_ids[:5],  # Limit
                    "url": f"https://www.uniprot.org/uniprotkb/{accession}",
                }
            )
            
        except Exception as e:
            logger.warning(f"[UniProt] Failed to parse entry: {e}")
            return None
