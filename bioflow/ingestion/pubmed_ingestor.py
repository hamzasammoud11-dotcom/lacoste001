"""
PubMed Ingestor - Biomedical Literature Ingestion
===================================================

Fetches abstracts from PubMed using NCBI E-utilities API.
https://www.ncbi.nlm.nih.gov/books/NBK25499/

Usage:
    from bioflow.ingestion import PubMedIngestor
    
    ingestor = PubMedIngestor(qdrant_service, obm_encoder)
    result = ingestor.ingest("EGFR lung cancer", limit=100)
"""

import logging
import requests
import xml.etree.ElementTree as ET
import re
from typing import Dict, Any, Optional, Generator
from datetime import datetime

from bioflow.ingestion.base_ingestor import BaseIngestor, DataRecord

logger = logging.getLogger(__name__)


class PubMedIngestor(BaseIngestor):
    """
    Ingestor for PubMed biomedical literature.
    
    Uses NCBI E-utilities:
    - esearch: Find article PMIDs matching query
    - efetch: Retrieve article details
    """
    
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    def __init__(
        self,
        qdrant_service,
        obm_encoder,
        collection: str = "bioflow_memory",
        batch_size: int = 50,
        rate_limit: float = 0.4,  # NCBI allows 3 requests/second without API key
        email: str = "bioflow@example.com",
        api_key: Optional[str] = None,
    ):
        """
        Initialize PubMed ingestor.
        
        Args:
            email: Email for NCBI (required by their policy)
            api_key: NCBI API key for higher rate limits (optional)
        """
        super().__init__(qdrant_service, obm_encoder, collection, batch_size, rate_limit)
        self.email = email
        self.api_key = api_key
    
    @property
    def source_name(self) -> str:
        return "pubmed"
    
    def _make_request(self, url: str, params: Dict[str, Any]) -> requests.Response:
        """Make a rate-limited request to NCBI."""
        self._rate_limit_wait()
        
        params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response
    
    def _search_pmids(self, query: str, limit: int) -> list:
        """Search for PMIDs matching query."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": min(limit, 10000),  # NCBI limit
            "retmode": "json",
            "sort": "relevance",
        }
        
        response = self._make_request(self.ESEARCH_URL, params)
        data = response.json()
        
        pmids = data.get("esearchresult", {}).get("idlist", [])
        logger.info(f"[PubMed] Found {len(pmids)} articles for query: {query}")
        
        return pmids[:limit]
    
    def _fetch_articles(self, pmids: list) -> str:
        """Fetch article details in XML format."""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
        }
        
        response = self._make_request(self.EFETCH_URL, params)
        return response.text
    
    def _parse_xml_articles(self, xml_text: str) -> Generator[Dict[str, Any], None, None]:
        """Parse PubMed XML into article dictionaries."""
        try:
            root = ET.fromstring(xml_text)
            
            for article in root.findall(".//PubmedArticle"):
                try:
                    # Extract PMID
                    pmid_elem = article.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else None
                    
                    if not pmid:
                        continue
                    
                    # Extract title
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else ""
                    
                    # Extract abstract
                    abstract_parts = []
                    for abstract_text in article.findall(".//AbstractText"):
                        label = abstract_text.get("Label", "")
                        text = abstract_text.text or ""
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)
                    abstract = " ".join(abstract_parts)
                    
                    # Extract publication date
                    pub_date = ""
                    date_elem = article.find(".//PubDate")
                    year_value = None
                    if date_elem is not None:
                        year = date_elem.find("Year")
                        month = date_elem.find("Month")
                        if year is not None:
                            pub_date = year.text
                            if month is not None:
                                pub_date = f"{year.text}-{month.text}"
                            try:
                                year_value = int(year.text)
                            except (TypeError, ValueError):
                                year_value = None
                    if year_value is None and pub_date:
                        match = re.match(r"(\\d{4})", pub_date)
                        if match:
                            year_value = int(match.group(1))
                    
                    # Extract authors
                    authors = []
                    for author in article.findall(".//Author"):
                        lastname = author.find("LastName")
                        forename = author.find("ForeName")
                        if lastname is not None:
                            name = lastname.text
                            if forename is not None:
                                name = f"{forename.text} {name}"
                            authors.append(name)
                    
                    # Extract journal
                    journal_elem = article.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else ""
                    
                    # Extract MeSH terms
                    mesh_terms = []
                    for mesh in article.findall(".//MeshHeading/DescriptorName"):
                        if mesh.text:
                            mesh_terms.append(mesh.text)
                    
                    yield {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "journal": journal,
                        "pub_date": pub_date,
                        "year": year_value,
                        "mesh_terms": mesh_terms,
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to parse article: {e}")
                    continue
                    
        except ET.ParseError as e:
            logger.error(f"XML parsing failed: {e}")
    
    def fetch_data(self, query: str, limit: int) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch PubMed articles matching query.
        
        Args:
            query: PubMed search query
            limit: Maximum articles to fetch
            
        Yields:
            Article dictionaries
        """
        # Step 1: Search for PMIDs
        pmids = self._search_pmids(query, limit)
        
        if not pmids:
            logger.warning(f"[PubMed] No results for query: {query}")
            return
        
        # Step 2: Fetch articles in batches
        batch_size = 100  # NCBI recommended batch size
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            
            try:
                xml_text = self._fetch_articles(batch_pmids)
                
                for article in self._parse_xml_articles(xml_text):
                    yield article
                    
            except Exception as e:
                logger.error(f"[PubMed] Batch fetch failed: {e}")
                continue
    
    def parse_record(self, raw_data: Dict[str, Any]) -> Optional[DataRecord]:
        """
        Parse a PubMed article into a DataRecord.
        
        Args:
            raw_data: Article dictionary
            
        Returns:
            DataRecord for text embedding
        """
        pmid = raw_data.get("pmid")
        title = raw_data.get("title", "")
        abstract = raw_data.get("abstract", "")
        
        # Skip articles without abstract
        if not abstract:
            logger.debug(f"[PubMed] Skipping {pmid}: no abstract")
            return None
        
        # Combine title and abstract for richer embedding
        content = f"{title}\n\n{abstract}"
        
        return DataRecord(
            id=f"pubmed:{pmid}",
            content=content,
            modality="text",
            metadata={
                "pmid": pmid,
                "title": title,
                "authors": raw_data.get("authors", []),
                "journal": raw_data.get("journal", ""),
                "pub_date": raw_data.get("pub_date", ""),
                "year": raw_data.get("year"),
                "mesh_terms": raw_data.get("mesh_terms", []),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
