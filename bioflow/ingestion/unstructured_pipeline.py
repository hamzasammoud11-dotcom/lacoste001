"""
Unstructured Data Pipeline - RAG for Lab Notes & Documents
============================================================

Handles ingestion and retrieval of unstructured biological data:
- Lab notes (text, handwritten → OCR)
- Paper abstracts and full texts
- PDF documents
- Protocol documents
- Experimental observations

This addresses the "Unstructured Data Pipeline" requirement:
- Parse/OCR documents
- Chunk into meaningful segments
- Index into vector database
- Enable retrieval-augmented generation (RAG)

Architecture:
1. Document Loader: Read various formats (PDF, DOCX, TXT, images)
2. OCR Engine: Extract text from images/scanned docs
3. Text Chunker: Split into semantically meaningful chunks
4. Metadata Extractor: Pull out key entities (molecules, targets, dates)
5. Embedder: Generate vectors for each chunk
6. Indexer: Store in Qdrant
"""

import logging
import re
import os
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Generator, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of text from a document."""
    id: str
    content: str
    chunk_index: int
    total_chunks: int
    source_file: str
    source_type: str  # pdf, docx, txt, image, lab_note
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    entities: Dict[str, List[str]] = field(default_factory=dict)  # Extracted entities
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "source_file": self.source_file,
            "source_type": self.source_type,
            "page_number": self.page_number,
            "metadata": self.metadata,
            "entities": self.entities,
        }


@dataclass
class LabNote:
    """Structured lab note record."""
    id: str
    title: str
    content: str
    date: Optional[str] = None
    author: Optional[str] = None
    experiment_id: Optional[str] = None
    protocol: Optional[str] = None
    observations: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    molecules: List[str] = field(default_factory=list)  # SMILES mentioned
    targets: List[str] = field(default_factory=list)  # Proteins/targets mentioned
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "date": self.date,
            "author": self.author,
            "experiment_id": self.experiment_id,
            "protocol": self.protocol,
            "observations": self.observations,
            "conclusions": self.conclusions,
            "molecules": self.molecules,
            "targets": self.targets,
            "measurements": self.measurements,
            "tags": self.tags,
        }


class UnstructuredDataPipeline:
    """
    Pipeline for processing unstructured biological documents.
    
    Supports:
    - PDF documents (papers, protocols)
    - Lab notebook images (with OCR)
    - Plain text files
    - JSON lab notes
    - Markdown documents
    
    Example:
        >>> pipeline = UnstructuredDataPipeline()
        >>> chunks = pipeline.process_document("lab_notes/exp001.pdf")
        >>> for chunk in chunks:
        ...     await qdrant_service.ingest(chunk.content, "text", chunk.metadata)
    """
    
    # Patterns for entity extraction
    SMILES_PATTERN = r'[A-Za-z0-9@+\-\[\]\(\)\\\/%=#$]{10,}'  # Simplified SMILES pattern
    PROTEIN_PATTERN = r'\b[A-Z][A-Z0-9]{2,}\b'  # Gene/protein names
    MEASUREMENT_PATTERN = r'(\d+(?:\.\d+)?)\s*(nM|µM|uM|mM|M|mg|µg|ug|g|kg|%|IC50|EC50|Ki|Kd)'
    DATE_PATTERN = r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}'
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        use_ocr: bool = True,
        extract_entities: bool = True,
    ):
        """
        Initialize pipeline.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            use_ocr: Enable OCR for images
            extract_entities: Extract molecules/proteins/measurements
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_ocr = use_ocr
        self.extract_entities = extract_entities
        
        self._pdf_available = False
        self._ocr_available = False
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check available processing libraries."""
        # Check PDF support
        try:
            import PyPDF2
            self._pdf_available = True
            logger.info("PyPDF2 available for PDF processing")
        except ImportError:
            try:
                import pdfplumber
                self._pdf_available = True
                logger.info("pdfplumber available for PDF processing")
            except ImportError:
                logger.warning("No PDF library available (install PyPDF2 or pdfplumber)")
        
        # Check OCR support
        if self.use_ocr:
            try:
                import pytesseract
                from PIL import Image
                self._ocr_available = True
                logger.info("Tesseract OCR available")
            except ImportError:
                logger.warning("OCR not available (install pytesseract and Pillow)")
    
    def process_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Process a document and return chunks.
        
        Args:
            file_path: Path to document
            metadata: Additional metadata to attach
            
        Returns:
            List of DocumentChunk
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        metadata = metadata or {}
        metadata["source_file"] = str(path.name)
        metadata["processed_at"] = datetime.utcnow().isoformat()
        
        extension = path.suffix.lower()
        
        if extension == ".pdf":
            return self._process_pdf(path, metadata)
        elif extension in [".txt", ".md", ".markdown"]:
            return self._process_text(path, metadata)
        elif extension == ".json":
            return self._process_json(path, metadata)
        elif extension in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            return self._process_image(path, metadata)
        else:
            logger.warning(f"Unsupported file type: {extension}")
            # Try as text
            return self._process_text(path, metadata)
    
    def process_lab_note(
        self,
        note: Union[str, Dict[str, Any], LabNote],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Process a lab note (text or structured).
        
        Args:
            note: Lab note content (string, dict, or LabNote)
            metadata: Additional metadata
            
        Returns:
            List of DocumentChunk
        """
        metadata = metadata or {}
        
        if isinstance(note, str):
            # Plain text note
            return self._chunk_text(
                text=note,
                source_file="lab_note",
                source_type="lab_note",
                metadata=metadata,
            )
        
        if isinstance(note, dict):
            note = LabNote(
                id=note.get("id", hashlib.md5(str(note).encode()).hexdigest()[:12]),
                title=note.get("title", "Untitled"),
                content=note.get("content", note.get("notes", "")),
                date=note.get("date"),
                author=note.get("author"),
                experiment_id=note.get("experiment_id"),
                protocol=note.get("protocol"),
                observations=note.get("observations", []),
                conclusions=note.get("conclusions", []),
                molecules=note.get("molecules", []),
                targets=note.get("targets", []),
                measurements=note.get("measurements", []),
                tags=note.get("tags", []),
            )
        
        # Build full text from lab note
        text_parts = [f"# {note.title}"]
        
        if note.date:
            text_parts.append(f"Date: {note.date}")
        if note.author:
            text_parts.append(f"Author: {note.author}")
        if note.experiment_id:
            text_parts.append(f"Experiment: {note.experiment_id}")
        
        text_parts.append("")
        text_parts.append(note.content)
        
        if note.protocol:
            text_parts.append(f"\n## Protocol\n{note.protocol}")
        
        if note.observations:
            text_parts.append("\n## Observations")
            for obs in note.observations:
                text_parts.append(f"- {obs}")
        
        if note.conclusions:
            text_parts.append("\n## Conclusions")
            for conc in note.conclusions:
                text_parts.append(f"- {conc}")
        
        if note.measurements:
            text_parts.append("\n## Measurements")
            for m in note.measurements:
                text_parts.append(f"- {m.get('name', 'Value')}: {m.get('value')} {m.get('unit', '')}")
        
        full_text = "\n".join(text_parts)
        
        # Enrich metadata
        enriched_metadata = {
            **metadata,
            "lab_note_id": note.id,
            "title": note.title,
            "date": note.date,
            "author": note.author,
            "experiment_id": note.experiment_id,
            "molecules": note.molecules,
            "targets": note.targets,
            "tags": note.tags,
        }
        
        return self._chunk_text(
            text=full_text,
            source_file=f"lab_note_{note.id}",
            source_type="lab_note",
            metadata=enriched_metadata,
        )
    
    def _process_pdf(
        self,
        path: Path,
        metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """Process PDF document."""
        if not self._pdf_available:
            logger.error("PDF processing not available")
            return []
        
        chunks = []
        full_text = []
        
        try:
            # Try PyPDF2 first
            try:
                import PyPDF2
                with open(path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text() or ""
                        full_text.append((page_num + 1, text))
            except ImportError:
                # Fall back to pdfplumber
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        full_text.append((page_num + 1, text))
            
            # Chunk with page awareness
            for page_num, text in full_text:
                page_chunks = self._chunk_text(
                    text=text,
                    source_file=str(path.name),
                    source_type="pdf",
                    metadata={**metadata, "page_number": page_num},
                )
                chunks.extend(page_chunks)
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
        
        return chunks
    
    def _process_text(
        self,
        path: Path,
        metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """Process plain text or markdown."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            source_type = "markdown" if path.suffix in [".md", ".markdown"] else "text"
            
            return self._chunk_text(
                text=text,
                source_file=str(path.name),
                source_type=source_type,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return []
    
    def _process_json(
        self,
        path: Path,
        metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """Process JSON file (structured data or array of notes)."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = []
            
            if isinstance(data, list):
                # Array of records
                for item in data:
                    if isinstance(item, dict):
                        item_chunks = self.process_lab_note(item, metadata)
                        chunks.extend(item_chunks)
            elif isinstance(data, dict):
                # Single record
                chunks = self.process_lab_note(data, metadata)
            
            return chunks
        except Exception as e:
            logger.error(f"JSON processing failed: {e}")
            return []
    
    def _process_image(
        self,
        path: Path,
        metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """Process image with OCR."""
        if not self._ocr_available:
            logger.warning("OCR not available, skipping image")
            return []
        
        try:
            import pytesseract
            from PIL import Image
            
            image = Image.open(path)
            text = pytesseract.image_to_string(image)
            
            if not text.strip():
                logger.warning(f"No text extracted from {path}")
                return []
            
            return self._chunk_text(
                text=text,
                source_file=str(path.name),
                source_type="ocr_image",
                metadata={**metadata, "ocr": True},
            )
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return []
    
    def _chunk_text(
        self,
        text: str,
        source_file: str,
        source_type: str,
        metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """
        Split text into chunks with overlap.
        
        Uses semantic chunking where possible (paragraphs, sections).
        """
        if not text.strip():
            return []
        
        chunks = []
        
        # Try to split on semantic boundaries first
        # Priority: sections > paragraphs > sentences > fixed size
        
        # Check for markdown/document sections
        if re.search(r'^#{1,6}\s', text, re.MULTILINE):
            segments = self._split_by_sections(text)
        elif '\n\n' in text:
            segments = self._split_by_paragraphs(text)
        else:
            segments = self._split_by_size(text)
        
        total_chunks = len(segments)
        
        for i, segment in enumerate(segments):
            chunk_id = hashlib.md5(
                f"{source_file}:{i}:{segment[:50]}".encode()
            ).hexdigest()[:16]
            
            # Extract entities if enabled
            entities = {}
            if self.extract_entities:
                entities = self._extract_entities(segment)
            
            chunks.append(DocumentChunk(
                id=chunk_id,
                content=segment.strip(),
                chunk_index=i,
                total_chunks=total_chunks,
                source_file=source_file,
                source_type=source_type,
                page_number=metadata.get("page_number"),
                metadata=metadata,
                entities=entities,
            ))
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Split by markdown-style sections."""
        sections = []
        current_section = []
        
        for line in text.split('\n'):
            if re.match(r'^#{1,6}\s', line) and current_section:
                # Start new section
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        # Further split large sections
        final_segments = []
        for section in sections:
            if len(section) > self.chunk_size * 2:
                final_segments.extend(self._split_by_size(section))
            else:
                final_segments.append(section)
        
        return final_segments
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split by paragraphs, merging small ones."""
        paragraphs = text.split('\n\n')
        segments = []
        current = []
        current_len = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if current_len + len(para) < self.chunk_size:
                current.append(para)
                current_len += len(para)
            else:
                if current:
                    segments.append('\n\n'.join(current))
                current = [para]
                current_len = len(para)
        
        if current:
            segments.append('\n\n'.join(current))
        
        return segments
    
    def _split_by_size(self, text: str) -> List[str]:
        """Fixed-size chunking with overlap."""
        segments = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 100 chars
                search_start = max(end - 100, start)
                sentence_end = None
                
                for pattern in ['. ', '.\n', '! ', '? ']:
                    idx = text.rfind(pattern, search_start, end)
                    if idx != -1 and (sentence_end is None or idx > sentence_end):
                        sentence_end = idx + 1
                
                if sentence_end:
                    end = sentence_end
            
            segments.append(text[start:end])
            start = end - self.chunk_overlap
        
        return segments
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract molecules, proteins, and measurements from text."""
        entities = {
            "molecules": [],
            "proteins": [],
            "measurements": [],
            "dates": [],
        }
        
        # Extract potential SMILES (very rough - would need validation)
        smiles_matches = re.findall(self.SMILES_PATTERN, text)
        for match in smiles_matches:
            # Basic validation: must have typical SMILES characters
            if any(c in match for c in ['=', '(', '[', 'c', 'C', 'N', 'O']):
                entities["molecules"].append(match)
        
        # Extract protein/gene names
        protein_matches = re.findall(self.PROTEIN_PATTERN, text)
        # Filter common false positives
        skip_words = {'THE', 'AND', 'FOR', 'WAS', 'WERE', 'HAS', 'HAD', 'ARE', 'NOT', 'BUT'}
        for match in protein_matches:
            if match not in skip_words and len(match) >= 3:
                entities["proteins"].append(match)
        
        # Extract measurements
        measurement_matches = re.findall(self.MEASUREMENT_PATTERN, text)
        for value, unit in measurement_matches:
            entities["measurements"].append(f"{value} {unit}")
        
        # Extract dates
        date_matches = re.findall(self.DATE_PATTERN, text)
        entities["dates"] = date_matches
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))[:10]  # Max 10 per category
        
        return entities


class RAGRetriever:
    """
    Retrieval-Augmented Generation support for unstructured data.
    
    Combines vector search with context augmentation for better
    question answering over lab notes and documents.
    """
    
    def __init__(
        self,
        qdrant_service,
        encoder,
        max_context_chunks: int = 5,
    ):
        """
        Initialize RAG retriever.
        
        Args:
            qdrant_service: QdrantService for vector search
            encoder: Encoder for query embedding
            max_context_chunks: Max chunks to include in context
        """
        self.qdrant = qdrant_service
        self.encoder = encoder
        self.max_context_chunks = max_context_chunks
    
    def retrieve_context(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks for a query.
        
        Args:
            query: User query
            filters: Optional filters (source_type, date range, etc.)
            min_score: Minimum similarity score
            
        Returns:
            List of relevant chunks with metadata
        """
        # Search for relevant chunks
        results = self.qdrant.search(
            query=query,
            modality="text",
            limit=self.max_context_chunks * 2,  # Get more, filter later
        )
        
        context_chunks = []
        for r in results:
            if r.score < min_score:
                continue
            
            # Apply filters
            if filters:
                if filters.get("source_type"):
                    if r.metadata.get("source_type") != filters["source_type"]:
                        continue
                if filters.get("experiment_id"):
                    if r.metadata.get("experiment_id") != filters["experiment_id"]:
                        continue
            
            context_chunks.append({
                "content": r.content,
                "score": r.score,
                "source": r.metadata.get("source_file", "unknown"),
                "source_type": r.metadata.get("source_type", "unknown"),
                "metadata": r.metadata,
            })
            
            if len(context_chunks) >= self.max_context_chunks:
                break
        
        return context_chunks
    
    def build_augmented_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Build an augmented prompt with retrieved context.
        
        This can be used with an LLM for question answering.
        """
        prompt_parts = [
            "Based on the following lab notes and documents, answer the question.\n",
            "---\n",
            "CONTEXT:\n",
        ]
        
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source", "Unknown")
            content = chunk.get("content", "")
            prompt_parts.append(f"\n[Source {i}: {source}]\n{content}\n")
        
        prompt_parts.append("---\n")
        prompt_parts.append(f"QUESTION: {query}\n")
        prompt_parts.append("ANSWER:")
        
        return "".join(prompt_parts)
