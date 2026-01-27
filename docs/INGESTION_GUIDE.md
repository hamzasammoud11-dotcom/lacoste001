# BioFlow Ingestion Guide (Phase 3)

This guide explains how to ingest data from **PubMed**, **UniProt**, and **ChEMBL** into Qdrant.

## 1) FastAPI Endpoints (Recommended)

### PubMed
`POST /api/ingest/pubmed`
```json
{
  "query": "EGFR lung cancer",
  "limit": 100,
  "batch_size": 50,
  "rate_limit": 0.4,
  "collection": "bioflow_memory",
  "email": "you@example.com",
  "api_key": "NCBI_API_KEY",
  "sync": false
}
```

### UniProt
`POST /api/ingest/uniprot`
```json
{
  "query": "EGFR AND organism_id:9606",
  "limit": 50,
  "batch_size": 50,
  "rate_limit": 0.2,
  "collection": "bioflow_memory",
  "sync": false
}
```

### ChEMBL
`POST /api/ingest/chembl`
```json
{
  "query": "EGFR",
  "limit": 30,
  "batch_size": 50,
  "rate_limit": 0.3,
  "collection": "bioflow_memory",
  "search_mode": "target",
  "sync": false
}
```

### All Sources
`POST /api/ingest/all`
```json
{
  "query": "EGFR lung cancer",
  "pubmed_limit": 100,
  "uniprot_limit": 50,
  "chembl_limit": 30,
  "batch_size": 50,
  "rate_limit": 0.3,
  "collection": "bioflow_memory",
  "sync": false
}
```

### Job Status
`GET /api/ingest/jobs/{job_id}`

## 2) Next.js Proxy Routes (Optional)
If you want to call the backend through Next.js:
```
/api/ingest/pubmed
/api/ingest/uniprot
/api/ingest/chembl
/api/ingest/all
/api/ingest/jobs/{job_id}
```

## 3) CLI Ingestion
```
python -m bioflow.ingestion.ingest_all --query "EGFR lung cancer" --limit 100
```

## 4) Environment Variables
- `INGEST_BATCH_SIZE`
- `PUBMED_RATE_LIMIT`
- `UNIPROT_RATE_LIMIT`
- `CHEMBL_RATE_LIMIT`
- `NCBI_EMAIL`
- `NCBI_API_KEY`
- `CHEMBL_SEARCH_MODE`

## 5) Recommended Minimums
- PubMed: 100 records
- UniProt: 50 records
- ChEMBL: 30 records

