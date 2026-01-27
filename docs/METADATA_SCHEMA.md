# BioFlow Metadata Schema (Phase 3)

All ingested items are stored in Qdrant with a **payload** that includes core provenance fields plus source‑specific metadata.

## Core Fields (all modalities)

| Field | Type | Description |
|------|------|-------------|
| `source` | string | Source name (`pubmed`, `uniprot`, `chembl`) |
| `source_id` | string | Source identifier (e.g., `pubmed:12345`) |
| `indexed_at` | string | ISO timestamp when ingested |
| `content` | string | Stored raw content (text, SMILES, or sequence) |
| `modality` | string | `text`, `molecule`, or `protein` |

## PubMed (text)

| Field | Type | Description |
|------|------|-------------|
| `pmid` | string | PubMed ID |
| `title` | string | Article title |
| `authors` | list[string] | Authors |
| `journal` | string | Journal name |
| `pub_date` | string | Publication date |
| `year` | number | Publication year |
| `mesh_terms` | list[string] | MeSH terms |
| `url` | string | PubMed URL |

## UniProt (protein)

| Field | Type | Description |
|------|------|-------------|
| `accession` | string | UniProt accession |
| `entry_name` | string | UniProt entry name |
| `protein_name` | string | Protein name |
| `gene_names` | list[string] | Gene names |
| `organism` | string | Scientific name |
| `organism_id` | string | Taxon ID |
| `function` | string | Function text (truncated) |
| `sequence_length` | number | Sequence length |
| `pdb_ids` | list[string] | PDB references |
| `url` | string | UniProt URL |

## ChEMBL (molecule)

| Field | Type | Description |
|------|------|-------------|
| `chembl_id` | string | ChEMBL molecule ID |
| `name` | string | Preferred name |
| `synonyms` | list[string] | Synonyms (limited) |
| `smiles` | string | Canonical SMILES |
| `inchi_key` | string | InChIKey |
| `molecular_weight` | number | Full molecular weight |
| `alogp` | number | ALogP |
| `hba` | number | H‑bond acceptors |
| `hbd` | number | H‑bond donors |
| `psa` | number | Polar surface area |
| `ro5_violations` | number | Rule‑of‑5 violations |
| `target_chembl_id` | string | Target ID (if available) |
| `activity_type` | string | Activity type (e.g., IC50) |
| `activity_value` | number | Activity value |
| `activity_units` | string | Activity units |
| `url` | string | ChEMBL URL |
