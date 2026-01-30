"""
Evidence Linking Module
========================

Provides source tracking and evidence linking for search results.
Generates DOI, UniProt, ChEMBL links from metadata.

CRITICAL for Use Case 4 (Requirement E):
- Every suggestion must cite which experiments/papers support it
- Evidence strength must be explicit (GOLD/STRONG/MODERATE/WEAK)
- Links must be clickable and traceable

Usage:
    from bioflow.search.evidence import EvidenceLinker
    
    linker = EvidenceLinker()
    enriched = linker.enrich(result)
    print(enriched.evidence_links)
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EvidenceStrength(Enum):
    """
    Evidence strength levels for scientific traceability.
    
    Use Case Requirement E: "evidence linking - showing why suggestions are made"
    """
    GOLD = "GOLD"           # Multiple independent validations (experimental + database + literature)
    STRONG = "STRONG"       # Experimental validation OR curated database entry
    MODERATE = "MODERATE"   # Literature support or computational prediction
    WEAK = "WEAK"           # Single source, limited metadata
    UNKNOWN = "UNKNOWN"     # Insufficient data


@dataclass
class EvidenceLink:
    """
    Represents a link to external evidence source.
    
    Every EvidenceLink should answer: "Why is this result trustworthy?"
    """
    source: str           # pubmed, uniprot, chembl, experiment, doi, etc.
    identifier: str       # PMID, accession, ChEMBL ID, experiment_id
    url: str              # Clickable URL to source
    label: str            # Human-readable description
    confidence: float = 1.0  # 0-1, how certain we are about this link
    evidence_type: str = "reference"  # reference, validation, supporting, contradicting


@dataclass 
class EnrichedResult:
    """
    Search result enriched with evidence links.
    
    Addresses Use Case 4 Requirements:
    - D.5: Scientific Traceability - link suggestions to evidence
    - E: Evidence Strength - explicit confidence levels
    """
    id: str
    score: float
    content: str
    modality: str
    metadata: Dict[str, Any]
    evidence_links: List[EvidenceLink] = field(default_factory=list)
    source_type: str = "unknown"
    citation: Optional[str] = None
    evidence_strength: EvidenceStrength = EvidenceStrength.UNKNOWN
    evidence_summary: str = ""  # Human-readable summary of why this is trustworthy


class EvidenceLinker:
    """
    Enriches search results with evidence links and citations.
    
    CRITICAL for Scientific Traceability (Use Case 4, Requirement E):
    - Every result must have traceable evidence
    - Evidence strength must be explicit
    - Supports experiments, papers, databases
    
    Supports:
    - PubMed (PMID → DOI, URL)
    - UniProt (Accession → URL, PDB links)
    - ChEMBL (Compound ID → URL, activity data)
    - Experiments (experiment_id → internal link, conditions, outcome)
    - Images (image_id → experiment linkage, image type)
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
        "pubchem": "https://pubchem.ncbi.nlm.nih.gov/compound/{id}",
        "clinicaltrials": "https://clinicaltrials.gov/study/{id}",
        "experiment": "/api/experiments/{id}",  # Internal experiment link
        "image": "/api/images/{id}",  # Internal image link
    }
    
    # Regex patterns for extracting IDs from text
    ID_PATTERNS = {
        "chembl": re.compile(r'CHEMBL\d+', re.IGNORECASE),
        "pubmed": re.compile(r'PMID[:\s]*(\d+)', re.IGNORECASE),
        "doi": re.compile(r'10\.\d{4,}/[^\s]+'),
        "pdb": re.compile(r'\b([1-9][A-Z0-9]{3})\b'),  # PDB codes like 1ABC
        "pubchem": re.compile(r'CID[:\s]*(\d+)', re.IGNORECASE),
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
            EnrichedResult with evidence links and strength assessment
        """
        metadata = result.get('metadata', result.get('payload', {}))
        source = metadata.get('source', '').lower()
        source_id = metadata.get('source_id', '')
        
        links = []
        citation = None
        source_type = source or "unknown"
        
        # Extract links based on source type
        if source == "pubmed":
            links.extend(self._extract_pubmed_links(metadata, source_id))
            citation = self._format_pubmed_citation(metadata)
        elif source == "uniprot":
            links.extend(self._extract_uniprot_links(metadata, source_id))
            citation = self._format_uniprot_citation(metadata)
        elif source == "chembl":
            links.extend(self._extract_chembl_links(metadata, source_id))
            citation = self._format_chembl_citation(metadata)
        elif source == "experiment":
            # NEW: Handle experiment evidence linking
            links.extend(self._extract_experiment_links(metadata, source_id))
            citation = self._format_experiment_citation(metadata)
        elif source == "image":
            # NEW: Handle image evidence linking
            links.extend(self._extract_image_links(metadata, source_id))
            citation = self._format_image_citation(metadata)
        
        # Also check for IDs in any field
        links.extend(self._extract_ids_from_metadata(metadata))
        
        # NEW: Extract IDs mentioned in text fields (description, notes, abstract)
        links.extend(self._extract_ids_from_text(metadata))
        
        # Deduplicate links
        seen = set()
        unique_links = []
        for link in links:
            key = (link.source, link.identifier)
            if key not in seen:
                seen.add(key)
                unique_links.append(link)
        
        # NEW: Calculate evidence strength
        evidence_strength, evidence_summary = self._calculate_evidence_strength(
            unique_links, metadata, source
        )
        
        return EnrichedResult(
            id=result.get('id', ''),
            score=result.get('score', 0),
            content=result.get('content', ''),
            modality=result.get('modality', 'unknown'),
            metadata=metadata,
            evidence_links=unique_links,
            source_type=source_type,
            citation=citation,
            evidence_strength=evidence_strength,
            evidence_summary=evidence_summary,
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

    # =========================================================================
    # NEW: Experiment Evidence Linking (Use Case 4 Requirement E)
    # =========================================================================
    
    def _extract_experiment_links(self, metadata: Dict, source_id: str) -> List[EvidenceLink]:
        """
        Extract evidence links from experimental data.
        
        CRITICAL for Scientific Traceability:
        - Links experiment to molecule/target
        - Includes experimental conditions
        - Shows outcome (success/failure)
        """
        links = []
        
        # Experiment ID link
        experiment_id = metadata.get('experiment_id') or source_id
        if experiment_id:
            links.append(EvidenceLink(
                source="experiment",
                identifier=experiment_id,
                url=self.URL_TEMPLATES["experiment"].format(id=experiment_id),
                label=f"Experiment: {experiment_id[:12]}...",
                confidence=1.0,
                evidence_type="validation"
            ))
        
        # Extract ChEMBL IDs mentioned in experiment description/notes
        description = metadata.get('description', '') + ' ' + metadata.get('notes', '')
        chembl_matches = self.ID_PATTERNS["chembl"].findall(description)
        for chembl_id in chembl_matches[:3]:  # Limit to 3
            links.append(EvidenceLink(
                source="chembl_compound",
                identifier=chembl_id,
                url=self.URL_TEMPLATES["chembl_compound"].format(id=chembl_id),
                label=f"ChEMBL: {chembl_id}",
                confidence=0.9,
                evidence_type="reference"
            ))
        
        # Extract PMIDs from abstract/notes
        abstract = metadata.get('abstract', '') + ' ' + metadata.get('notes', '')
        pmid_matches = self.ID_PATTERNS["pubmed"].findall(abstract)
        for pmid in pmid_matches[:3]:
            links.append(EvidenceLink(
                source="pubmed",
                identifier=pmid,
                url=self.URL_TEMPLATES["pubmed"].format(id=pmid),
                label=f"PubMed: {pmid}",
                confidence=0.95,
                evidence_type="supporting"
            ))
        
        # Target link if available
        target = metadata.get('target') or metadata.get('target_id')
        if target:
            # Try to find UniProt ID pattern
            if re.match(r'^[A-Z][0-9][A-Z0-9]{3}[0-9]$', str(target)):
                links.append(EvidenceLink(
                    source="uniprot",
                    identifier=target,
                    url=self.URL_TEMPLATES["uniprot"].format(id=target),
                    label=f"Target: {target}",
                    confidence=0.9,
                    evidence_type="reference"
                ))
        
        # Molecule/compound link if SMILES available
        smiles = metadata.get('molecule') or metadata.get('smiles')
        if smiles and metadata.get('pubchem_cid'):
            links.append(EvidenceLink(
                source="pubchem",
                identifier=str(metadata['pubchem_cid']),
                url=self.URL_TEMPLATES["pubchem"].format(id=metadata['pubchem_cid']),
                label=f"PubChem: {metadata['pubchem_cid']}",
                confidence=0.95,
                evidence_type="reference"
            ))
        
        return links
    
    def _format_experiment_citation(self, metadata: Dict) -> Optional[str]:
        """
        Format an experiment citation with reproducibility info.
        
        Includes: experiment type, outcome, target, conditions summary
        """
        experiment_id = metadata.get('experiment_id', '')
        exp_type = metadata.get('experiment_type', 'assay')
        outcome = metadata.get('outcome', '')
        target = metadata.get('target', '')
        molecule = metadata.get('molecule_name', metadata.get('molecule', ''))
        
        if not experiment_id:
            return None
        
        # Build citation
        parts = []
        
        # Experiment type and outcome
        outcome_emoji = {"success": "✅", "failure": "❌", "partial": "⚠️"}.get(outcome, "🔬")
        parts.append(f"{outcome_emoji} {exp_type.replace('_', ' ').title()}")
        
        # Target info
        if target:
            parts.append(f"Target: {target}")
        
        # Molecule info (truncated)
        if molecule:
            mol_short = molecule[:30] + "..." if len(molecule) > 30 else molecule
            parts.append(f"Compound: {mol_short}")
        
        # Measurements summary
        measurements = metadata.get('measurements', [])
        if measurements and isinstance(measurements, list) and len(measurements) > 0:
            m = measurements[0]
            if isinstance(m, dict):
                parts.append(f"{m.get('name', 'Value')}: {m.get('value', 'N/A')} {m.get('unit', '')}")
        
        # Quality score
        quality = metadata.get('quality_score')
        if quality:
            parts.append(f"Quality: {float(quality):.0%}")
        
        citation = " | ".join(parts)
        citation += f" [Exp: {experiment_id[:8]}]"
        
        return citation
    
    def _extract_image_links(self, metadata: Dict, source_id: str) -> List[EvidenceLink]:
        """Extract evidence links from image metadata."""
        links = []
        
        # Image ID link
        image_id = metadata.get('image_id') or source_id
        if image_id:
            links.append(EvidenceLink(
                source="image",
                identifier=image_id,
                url=self.URL_TEMPLATES["image"].format(id=image_id),
                label=f"Image: {image_id[:12]}...",
                confidence=1.0,
                evidence_type="supporting"
            ))
        
        # Linked experiment
        experiment_id = metadata.get('experiment_id')
        if experiment_id:
            links.append(EvidenceLink(
                source="experiment",
                identifier=experiment_id,
                url=self.URL_TEMPLATES["experiment"].format(id=experiment_id),
                label=f"Source Experiment: {experiment_id[:12]}...",
                confidence=0.95,
                evidence_type="validation"
            ))
        
        # PDB link for protein images
        pdb_id = metadata.get('pdb_id')
        if pdb_id:
            links.append(EvidenceLink(
                source="pdb",
                identifier=pdb_id,
                url=self.URL_TEMPLATES["pdb"].format(id=pdb_id),
                label=f"PDB: {pdb_id}",
                confidence=1.0,
                evidence_type="reference"
            ))
        
        return links
    
    def _format_image_citation(self, metadata: Dict) -> Optional[str]:
        """Format an image citation."""
        image_type = metadata.get('image_type', 'image')
        description = metadata.get('description', metadata.get('title', ''))
        source = metadata.get('source', '')
        
        parts = []
        
        # Image type with emoji
        type_emoji = {
            "gel": "🧬",
            "western_blot": "🧬",
            "microscopy": "🔬",
            "fluorescence": "🔬",
            "spectra": "📊",
            "xray": "💎",
            "pdb": "🔷",
        }
        emoji = type_emoji.get(image_type.lower(), "🖼️")
        parts.append(f"{emoji} {image_type.replace('_', ' ').title()}")
        
        if description:
            desc_short = description[:50] + "..." if len(description) > 50 else description
            parts.append(desc_short)
        
        if source:
            parts.append(f"Source: {source}")
        
        return " | ".join(parts) if parts else None
    
    def _extract_ids_from_text(self, metadata: Dict) -> List[EvidenceLink]:
        """
        Extract database IDs from text fields (description, notes, abstract).
        
        This catches references that weren't explicitly tagged in metadata.
        """
        links = []
        
        # Combine all text fields
        text_fields = ['description', 'notes', 'abstract', 'protocol', 'content']
        combined_text = ' '.join(str(metadata.get(f, '')) for f in text_fields)
        
        if not combined_text.strip():
            return links
        
        # Extract ChEMBL IDs
        chembl_ids = set(self.ID_PATTERNS["chembl"].findall(combined_text))
        for chembl_id in list(chembl_ids)[:3]:
            links.append(EvidenceLink(
                source="chembl_compound",
                identifier=chembl_id.upper(),
                url=self.URL_TEMPLATES["chembl_compound"].format(id=chembl_id.upper()),
                label=f"ChEMBL: {chembl_id.upper()}",
                confidence=0.85,
                evidence_type="reference"
            ))
        
        # Extract DOIs
        dois = set(self.ID_PATTERNS["doi"].findall(combined_text))
        for doi in list(dois)[:2]:
            links.append(EvidenceLink(
                source="doi",
                identifier=doi,
                url=self.URL_TEMPLATES["doi"].format(id=doi),
                label=f"DOI: {doi}",
                confidence=0.95,
                evidence_type="supporting"
            ))
        
        # Extract PubChem CIDs
        cids = set(self.ID_PATTERNS["pubchem"].findall(combined_text))
        for cid in list(cids)[:2]:
            links.append(EvidenceLink(
                source="pubchem",
                identifier=cid,
                url=self.URL_TEMPLATES["pubchem"].format(id=cid),
                label=f"PubChem: {cid}",
                confidence=0.85,
                evidence_type="reference"
            ))
        
        return links
    
    def _calculate_evidence_strength(
        self, 
        links: List[EvidenceLink], 
        metadata: Dict, 
        source: str
    ) -> Tuple[EvidenceStrength, str]:
        """
        Calculate overall evidence strength for a result.
        
        Returns (EvidenceStrength enum, human-readable summary)
        
        Scoring:
        - GOLD (15+): Multiple independent validations
        - STRONG (10-14): Experimental validation or curated database
        - MODERATE (5-9): Literature support or single database
        - WEAK (2-4): Limited evidence
        - UNKNOWN (0-1): Insufficient data
        """
        score = 0
        reasons = []
        
        # Score based on source type
        source_scores = {
            "experiment": 6,
            "chembl": 5,
            "drugbank": 5,
            "pubmed": 4,
            "uniprot": 4,
            "pdb": 3,
            "pubchem": 2,
        }
        if source in source_scores:
            score += source_scores[source]
            reasons.append(f"Source: {source}")
        
        # Score based on evidence links
        validation_links = sum(1 for l in links if l.evidence_type == "validation")
        reference_links = sum(1 for l in links if l.evidence_type == "reference")
        supporting_links = sum(1 for l in links if l.evidence_type == "supporting")
        
        score += validation_links * 3
        score += reference_links * 2
        score += supporting_links * 1
        
        if validation_links > 0:
            reasons.append(f"{validation_links} validation(s)")
        if reference_links > 0:
            reasons.append(f"{reference_links} reference(s)")
        
        # Score based on experimental outcome
        outcome = metadata.get('outcome', '')
        if outcome == 'success':
            score += 4
            reasons.append("Successful experiment")
        elif outcome == 'failure':
            score += 2
            reasons.append("Failed experiment (negative data)")
        elif outcome:
            score += 1
        
        # Score based on activity data
        if metadata.get('activity_type') or metadata.get('measurements'):
            score += 2
            reasons.append("Activity data available")
        
        # Score based on quality
        quality = metadata.get('quality_score')
        if quality:
            score += int(float(quality) * 3)
            if float(quality) >= 0.8:
                reasons.append(f"High quality ({float(quality):.0%})")
        
        # Determine strength level
        if score >= 15:
            strength = EvidenceStrength.GOLD
        elif score >= 10:
            strength = EvidenceStrength.STRONG
        elif score >= 5:
            strength = EvidenceStrength.MODERATE
        elif score >= 2:
            strength = EvidenceStrength.WEAK
        else:
            strength = EvidenceStrength.UNKNOWN
        
        # Build summary
        summary = f"[{strength.value}] " + ", ".join(reasons[:3]) if reasons else f"[{strength.value}] Limited evidence"
        
        return strength, summary
