"""
Evidence Linking Module
========================

Provides source tracking and evidence linking for search results.
Generates DOI, UniProt, ChEMBL links from metadata.

Usage:
    from bioflow.search.evidence import EvidenceLinker
    
    linker = EvidenceLinker()
    enriched = linker.enrich(result)
    print(enriched.evidence_links)
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvidenceLink:
    """Represents a link to external evidence source."""
    source: str  # pubmed, uniprot, chembl, doi, etc.
    identifier: str  # PMID, accession, ChEMBL ID
    url: str
    label: str
    confidence: float = 1.0  # 0-1, how certain we are about this link


@dataclass 
class EnrichedResult:
    """Search result enriched with evidence links."""
    id: str
    score: float
    content: str
    modality: str
    metadata: Dict[str, Any]
    evidence_links: List[EvidenceLink] = field(default_factory=list)
    source_type: str = "unknown"
    citation: Optional[str] = None


class EvidenceLinker:
    """
    Enriches search results with evidence links and citations.
    
    Supports:
    - PubMed (PMID → DOI, URL)
    - UniProt (Accession → URL, PDB links)
    - ChEMBL (Compound ID → URL, activity data)
    """
    
    # URL templates for external databases
    URL_TEMPLATES = {
        "pubmed": "https://pubmed.ncbi.nlm.nih.gov/{id}/",
        "doi": "https://doi.org/{id}",
        "uniprot": "https://www.uniprot.org/uniprotkb/{id}",
        "pdb": "https://www.rcsb.org/structure/{id}",
        "chembl_compound": "https://www.ebi.ac.uk/chembl/compound_report_card/{id}/",
        "chembl_target": "https://www.ebi.ac.uk/chembl/target_report_card/{id}/",
        "drugbank": "https://go.drugbank.com/drugs/{id}",
        "ncbi_gene": "https://www.ncbi.nlm.nih.gov/gene/{id}",
    }
    
    def __init__(self):
        """Initialize evidence linker."""
        pass
    
    def enrich(self, result: Dict[str, Any]) -> EnrichedResult:
        """
        Enrich a search result with evidence links.
        
        Args:
            result: Search result with metadata
            
        Returns:
            EnrichedResult with evidence links
        """
        metadata = result.get('metadata', result.get('payload', {}))
        source = metadata.get('source', '').lower()
        source_id = metadata.get('source_id', '')
        
        links = []
        citation = None
        source_type = source or "unknown"
        
        # Extract links based on source
        if source == "pubmed":
            links.extend(self._extract_pubmed_links(metadata, source_id))
            citation = self._format_pubmed_citation(metadata)
        elif source == "uniprot":
            links.extend(self._extract_uniprot_links(metadata, source_id))
            citation = self._format_uniprot_citation(metadata)
        elif source == "chembl":
            links.extend(self._extract_chembl_links(metadata, source_id))
            citation = self._format_chembl_citation(metadata)
        
        # Also check for IDs in any field
        links.extend(self._extract_ids_from_metadata(metadata))
        
        # Deduplicate links
        seen = set()
        unique_links = []
        for link in links:
            key = (link.source, link.identifier)
            if key not in seen:
                seen.add(key)
                unique_links.append(link)
        
        return EnrichedResult(
            id=result.get('id', ''),
            score=result.get('score', 0),
            content=result.get('content', ''),
            modality=result.get('modality', 'unknown'),
            metadata=metadata,
            evidence_links=unique_links,
            source_type=source_type,
            citation=citation,
        )
    
    def enrich_batch(self, results: List[Dict[str, Any]]) -> List[EnrichedResult]:
        """Enrich multiple results."""
        return [self.enrich(r) for r in results]
    
    def _extract_pubmed_links(self, metadata: Dict, source_id: str) -> List[EvidenceLink]:
        """Extract links from PubMed metadata."""
        links = []
        
        # PMID link
        pmid = metadata.get('pmid') or self._extract_id(source_id, 'pubmed:')
        if pmid:
            links.append(EvidenceLink(
                source="pubmed",
                identifier=pmid,
                url=self.URL_TEMPLATES["pubmed"].format(id=pmid),
                label=f"PubMed: {pmid}"
            ))
        
        # DOI if available
        doi = metadata.get('doi')
        if doi:
            links.append(EvidenceLink(
                source="doi",
                identifier=doi,
                url=self.URL_TEMPLATES["doi"].format(id=doi),
                label=f"DOI: {doi}"
            ))
        
        return links
    
    def _extract_uniprot_links(self, metadata: Dict, source_id: str) -> List[EvidenceLink]:
        """Extract links from UniProt metadata."""
        links = []
        
        # UniProt accession
        accession = metadata.get('accession') or self._extract_id(source_id, 'uniprot:')
        if accession:
            links.append(EvidenceLink(
                source="uniprot",
                identifier=accession,
                url=self.URL_TEMPLATES["uniprot"].format(id=accession),
                label=f"UniProt: {accession}"
            ))
        
        # PDB structures
        pdb_ids = metadata.get('pdb_ids', [])
        if isinstance(pdb_ids, str):
            pdb_ids = [pdb_ids]
        for pdb_id in pdb_ids[:3]:  # Limit to 3
            links.append(EvidenceLink(
                source="pdb",
                identifier=pdb_id,
                url=self.URL_TEMPLATES["pdb"].format(id=pdb_id),
                label=f"PDB: {pdb_id}"
            ))
        
        # Gene ID
        gene_id = metadata.get('gene_id')
        if gene_id:
            links.append(EvidenceLink(
                source="ncbi_gene",
                identifier=str(gene_id),
                url=self.URL_TEMPLATES["ncbi_gene"].format(id=gene_id),
                label=f"Gene: {gene_id}"
            ))
        
        return links
    
    def _extract_chembl_links(self, metadata: Dict, source_id: str) -> List[EvidenceLink]:
        """Extract links from ChEMBL metadata."""
        links = []
        
        # ChEMBL compound ID
        chembl_id = metadata.get('chembl_id') or self._extract_id(source_id, 'chembl:')
        if chembl_id:
            links.append(EvidenceLink(
                source="chembl_compound",
                identifier=chembl_id,
                url=self.URL_TEMPLATES["chembl_compound"].format(id=chembl_id),
                label=f"ChEMBL: {chembl_id}"
            ))
        
        # Target ID if available
        target_id = metadata.get('target_chembl_id')
        if target_id:
            links.append(EvidenceLink(
                source="chembl_target",
                identifier=target_id,
                url=self.URL_TEMPLATES["chembl_target"].format(id=target_id),
                label=f"Target: {target_id}"
            ))
        
        # DrugBank if available
        drugbank_id = metadata.get('drugbank_id')
        if drugbank_id:
            links.append(EvidenceLink(
                source="drugbank",
                identifier=drugbank_id,
                url=self.URL_TEMPLATES["drugbank"].format(id=drugbank_id),
                label=f"DrugBank: {drugbank_id}"
            ))
        
        return links
    
    def _extract_ids_from_metadata(self, metadata: Dict) -> List[EvidenceLink]:
        """Extract any IDs from metadata fields."""
        links = []
        
        # Look for common ID patterns
        id_patterns = {
            'pmid': ('pubmed', 'pubmed'),
            'pubmed_id': ('pubmed', 'pubmed'),
            'uniprot_id': ('uniprot', 'uniprot'),
            'pdb_id': ('pdb', 'pdb'),
            'doi': ('doi', 'doi'),
        }
        
        for field, (source, template_key) in id_patterns.items():
            value = metadata.get(field)
            if value and template_key in self.URL_TEMPLATES:
                links.append(EvidenceLink(
                    source=source,
                    identifier=str(value),
                    url=self.URL_TEMPLATES[template_key].format(id=value),
                    label=f"{source.upper()}: {value}"
                ))
        
        return links
    
    def _extract_id(self, source_id: str, prefix: str) -> Optional[str]:
        """Extract ID from prefixed source_id (e.g., 'pubmed:12345' → '12345')."""
        if source_id.startswith(prefix):
            return source_id[len(prefix):]
        return None
    
    def _format_pubmed_citation(self, metadata: Dict) -> Optional[str]:
        """Format a PubMed citation."""
        authors = metadata.get('authors', [])
        title = metadata.get('title', '')
        journal = metadata.get('journal', '')
        year = metadata.get('year', metadata.get('pub_date', ''))
        pmid = metadata.get('pmid', '')
        
        if not title:
            return None
        
        # Format: Authors (Year). Title. Journal. PMID: XXX
        author_str = authors[0] if authors else "Unknown"
        if len(authors) > 1:
            author_str += " et al."
        
        citation = f"{author_str}"
        if year:
            citation += f" ({year})"
        citation += f". {title}"
        if journal:
            citation += f" {journal}."
        if pmid:
            citation += f" PMID: {pmid}"
        
        return citation
    
    def _format_uniprot_citation(self, metadata: Dict) -> Optional[str]:
        """Format a UniProt citation."""
        accession = metadata.get('accession', '')
        protein_name = metadata.get('protein_name', '')
        gene_names = metadata.get('gene_names', [])
        organism = metadata.get('organism', '')
        
        if not accession:
            return None
        
        citation = f"{protein_name or 'Unknown protein'}"
        if gene_names:
            if isinstance(gene_names, list):
                citation += f" ({', '.join(gene_names[:2])})"
            else:
                citation += f" ({gene_names})"
        if organism:
            citation += f" - {organism}"
        citation += f" [UniProt: {accession}]"
        
        return citation
    
    def _format_chembl_citation(self, metadata: Dict) -> Optional[str]:
        """Format a ChEMBL citation."""
        chembl_id = metadata.get('chembl_id', '')
        name = metadata.get('pref_name', metadata.get('molecule_name', ''))
        smiles = metadata.get('smiles', '')
        
        if not chembl_id:
            return None
        
        citation = name or chembl_id
        if smiles:
            # Truncate long SMILES
            short_smiles = smiles[:50] + "..." if len(smiles) > 50 else smiles
            citation += f" ({short_smiles})"
        citation += f" [ChEMBL: {chembl_id}]"
        
        return citation
