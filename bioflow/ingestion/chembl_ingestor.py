"""
ChEMBL Ingestor - Small Molecule Ingestion
============================================

Fetches molecules from ChEMBL using their REST API.
https://www.ebi.ac.uk/chembl/api/data/docs

Usage:
    from bioflow.ingestion import ChEMBLIngestor
    
    ingestor = ChEMBLIngestor(qdrant_service, obm_encoder)
    result = ingestor.ingest("EGFR", limit=50)  # Search by target name
"""

import logging
import requests
from typing import Dict, Any, Optional, Generator, List

from bioflow.ingestion.base_ingestor import BaseIngestor, DataRecord

logger = logging.getLogger(__name__)


class ChEMBLIngestor(BaseIngestor):
    """
    Ingestor for ChEMBL small molecules.
    
    Supports two modes:
    1. Search by target name → get active compounds
    2. Search by molecule name/synonym
    """
    
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
    
    def __init__(
        self,
        qdrant_service,
        obm_encoder,
        collection: str = "bioflow_memory",
        batch_size: int = 50,
        rate_limit: float = 0.3,
        search_mode: str = "target",  # "target" or "molecule"
    ):
        """
        Initialize ChEMBL ingestor.
        
        Args:
            search_mode: "target" to find compounds active against a target,
                        "molecule" to search molecules by name
        """
        super().__init__(qdrant_service, obm_encoder, collection, batch_size, rate_limit)
        self.search_mode = search_mode
    
    @property
    def source_name(self) -> str:
        return "chembl"
    
    def _search_targets(self, query: str, limit: int = 10) -> List[str]:
        """Search for target ChEMBL IDs matching query."""
        self._rate_limit_wait()
        
        url = f"{self.BASE_URL}/target/search.json"
        params = {
            "q": query,
            "limit": limit,
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            targets = data.get("targets", [])
            target_ids = [t.get("target_chembl_id") for t in targets if t.get("target_chembl_id")]
            
            logger.info(f"[ChEMBL] Found {len(target_ids)} targets for: {query}")
            return target_ids
            
        except Exception as e:
            logger.error(f"[ChEMBL] Target search failed: {e}")
            return []
    
    def _get_activities_for_target(self, target_id: str, limit: int) -> Generator[Dict[str, Any], None, None]:
        """Get bioactivity data for a target."""
        url = f"{self.BASE_URL}/activity.json"
        
        params = {
            "target_chembl_id": target_id,
            "limit": min(limit, 1000),
            "offset": 0,
        }
        
        fetched = 0
        
        while fetched < limit:
            try:
                self._rate_limit_wait()
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                activities = data.get("activities", [])
                
                if not activities:
                    break
                
                for activity in activities:
                    if fetched >= limit:
                        break
                    yield activity
                    fetched += 1
                
                # Check for more pages
                if data.get("page_meta", {}).get("next"):
                    params["offset"] += params["limit"]
                else:
                    break
                    
            except Exception as e:
                logger.error(f"[ChEMBL] Activity fetch failed: {e}")
                break
    
    def _get_molecule_details(self, chembl_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed molecule information."""
        self._rate_limit_wait()
        
        url = f"{self.BASE_URL}/molecule/{chembl_id}.json"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.debug(f"[ChEMBL] Molecule fetch failed for {chembl_id}: {e}")
            return None
    
    def _search_molecules(self, query: str, limit: int) -> Generator[Dict[str, Any], None, None]:
        """Search molecules by name/synonym."""
        url = f"{self.BASE_URL}/molecule/search.json"
        
        params = {
            "q": query,
            "limit": min(limit, 1000),
        }
        
        try:
            self._rate_limit_wait()
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            molecules = data.get("molecules", [])
            
            for mol in molecules[:limit]:
                yield mol
                
        except Exception as e:
            logger.error(f"[ChEMBL] Molecule search failed: {e}")
    
    def fetch_data(self, query: str, limit: int) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch molecules from ChEMBL.
        
        Args:
            query: Target name or molecule name depending on search_mode
            limit: Maximum molecules to fetch
            
        Yields:
            Molecule data dictionaries
        """
        if self.search_mode == "target":
            # Find targets matching query
            target_ids = self._search_targets(query, limit=5)
            
            if not target_ids:
                logger.warning(f"[ChEMBL] No targets found for: {query}")
                return
            
            # Get activities for each target
            seen_molecules = set()
            fetched = 0
            
            for target_id in target_ids:
                if fetched >= limit:
                    break
                
                for activity in self._get_activities_for_target(target_id, limit - fetched):
                    mol_id = activity.get("molecule_chembl_id")
                    
                    # Skip duplicates
                    if mol_id in seen_molecules:
                        continue
                    seen_molecules.add(mol_id)
                    
                    # Enrich with molecule details
                    mol_details = self._get_molecule_details(mol_id)
                    
                    if mol_details:
                        mol_details["activity"] = activity
                        mol_details["target_chembl_id"] = target_id
                        yield mol_details
                        fetched += 1
                    
                    if fetched >= limit:
                        break
        else:
            # Direct molecule search
            yield from self._search_molecules(query, limit)
    
    def parse_record(self, raw_data: Dict[str, Any]) -> Optional[DataRecord]:
        """
        Parse a ChEMBL molecule into a DataRecord.
        
        Args:
            raw_data: ChEMBL molecule JSON
            
        Returns:
            DataRecord for molecule embedding
        """
        try:
            mol_id = raw_data.get("molecule_chembl_id", "")
            
            if not mol_id:
                return None
            
            # Get SMILES
            structures = raw_data.get("molecule_structures", {}) or {}
            smiles = structures.get("canonical_smiles", "")
            
            if not smiles:
                logger.debug(f"[ChEMBL] Skipping {mol_id}: no SMILES")
                return None
            
            # Get molecule properties
            properties = raw_data.get("molecule_properties", {}) or {}
            
            # Get preferred name
            pref_name = raw_data.get("pref_name", "")
            
            # Get synonyms
            synonyms = []
            for syn in raw_data.get("molecule_synonyms", []) or []:
                if syn.get("molecule_synonym"):
                    synonyms.append(syn["molecule_synonym"])
            
            # Get activity data if available
            activity = raw_data.get("activity", {})
            
            return DataRecord(
                id=f"chembl:{mol_id}",
                content=smiles,
                modality="molecule",
                metadata={
                    "chembl_id": mol_id,
                    "name": pref_name,
                    "synonyms": synonyms[:5],
                    "smiles": smiles,
                    "inchi_key": structures.get("standard_inchi_key", ""),
                    "molecular_weight": properties.get("full_mwt"),
                    "alogp": properties.get("alogp"),
                    "hba": properties.get("hba"),
                    "hbd": properties.get("hbd"),
                    "psa": properties.get("psa"),
                    "ro5_violations": properties.get("num_ro5_violations"),
                    # Activity data
                    "target_chembl_id": raw_data.get("target_chembl_id", ""),
                    "activity_type": activity.get("standard_type", ""),
                    "activity_value": activity.get("standard_value"),
                    "activity_units": activity.get("standard_units", ""),
                    "url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{mol_id}/",
                }
            )
            
        except Exception as e:
            logger.warning(f"[ChEMBL] Failed to parse molecule: {e}")
            return None
