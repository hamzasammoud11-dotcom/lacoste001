import { API_CONFIG } from '@/config/api.config';
import { DataResponse } from '@/schemas/data';
import { ExplorerRequestSchema, ExplorerResponse } from '@/schemas/explorer';
import { PredictionResponse } from '@/schemas/prediction';
import { SearchResult } from '@/schemas/search';
import { EmbeddingPoint } from '@/schemas/visualization';
import { Candidate,WorkflowResult } from '@/schemas/workflow';
import { Molecule, Protein } from '@/types/visualization';

const API_BASE = API_CONFIG.baseUrl;

// --- UTILS ---

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
    const response = await fetch(url, options);
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.error || error.detail || `HTTP ${response.status}`);
    }
    return response.json();
}

async function fetchText(url: string, options?: RequestInit): Promise<string> {
    const response = await fetch(url, options);
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.error || error.detail || `HTTP ${response.status}`);
    }
    return response.text();
}

// --- DATA / STATS ---

/**
 * Fetches dashboard statistics and dataset information.
 */
export async function getStats(): Promise<DataResponse> {
    return fetchJson<DataResponse>(`${API_BASE}/api/stats`, {
        next: { revalidate: 60 },
    });
}

// --- EXPLORER ---

/**
 * Fetches 3D points for the embedding explorer.
 */
export async function getExplorerPoints(
    dataset?: string,
    view?: string,
    colorBy?: string,
): Promise<ExplorerResponse> {
    const result = ExplorerRequestSchema.safeParse({ dataset, view, colorBy });

    if (!result.success) {
        throw new Error('Invalid parameters');
    }

    const apiView =
        view === 'UMAP'
            ? 'combined'
            : view === 'PCA-Drug'
                ? 'drug'
                : view === 'PCA-Target'
                    ? 'target'
                    : 'combined';

    return fetchJson<ExplorerResponse>(
        `${API_BASE}/api/points?limit=500&view=${apiView}`,
        {
            next: { revalidate: 0 },
            signal: AbortSignal.timeout(5000),
        },
    );
}

// --- MOLECULES / PROTEINS ---

interface MoleculesResponse {
    molecules: Molecule[];
    total: number;
    limit: number;
    offset: number;
}

export async function getMolecules(): Promise<Molecule[]> {
    const response = await fetchJson<MoleculesResponse>(`${API_BASE}/api/molecules`);
    return response.molecules || [];
}

export async function getMolecule(id: string): Promise<Molecule> {
    return fetchJson<Molecule>(`${API_BASE}/api/molecules/${id}`);
}

export async function getMoleculeSDF(id: string): Promise<string> {
    return fetchText(`${API_BASE}/api/molecules/${id}/sdf`);
}

export function getMoleculeSdfUrl(id: string): string {
    return `${API_BASE}/api/molecules/${id}/sdf`;
}

interface ProteinsResponse {
    proteins: Protein[];
    total: number;
    limit: number;
    offset: number;
}

export async function getProteins(): Promise<Protein[]> {
    const response = await fetchJson<ProteinsResponse>(`${API_BASE}/api/proteins`);
    return response.proteins || [];
}

export async function getProtein(id: string): Promise<Protein> {
    return fetchJson<Protein>(`${API_BASE}/api/proteins/${id}`);
}

export async function getProteinPDB(id: string): Promise<string> {
    return fetchText(`${API_BASE}/api/proteins/${id}/pdb`);
}

export function getProteinPdbUrl(id: string): string {
    // If it's a 4-letter PDB ID, use RCSB, otherwise use local API
    if (id.length === 4) {
        return `https://files.rcsb.org/download/${id.toUpperCase()}.pdb`;
    }
    return `${API_BASE}/api/proteins/${id}/pdb`;
}

// --- DISCOVERY & SEARCH ---

/**
 * General search across modalities.
 */
export async function search(params: {
    query: string;
    type?: string;
    limit?: number;
    dataset?: string;
    top_k?: number;
    use_mmr?: boolean;
}): Promise<{ results: SearchResult[] }> {
    return fetchJson(`${API_BASE}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
    });
}

/**
 * Fetches embeddings for a query.
 */
export async function getEmbeddings(
    query: string,
    method: string = 'pca',
    limit: number = 50,
): Promise<{ points: EmbeddingPoint[] }> {
    return fetchJson(
        `${API_BASE}/api/explorer/embeddings?query=${encodeURIComponent(query)}&method=${method}&limit=${limit}`,
    );
}

/**
 * Predicts binding affinity between a drug and a target.
 */
export async function predict(drug_smiles: string, target_sequence: string): Promise<PredictionResponse> {
    return fetchJson(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ drug_smiles, target_sequence }),
    });
}

// --- WORKFLOW ---

/**
 * Executes a discovery workflow.
 */
export async function runWorkflow(params: {
    query: string;
    num_candidates: number;
    top_k: number;
}): Promise<WorkflowResult & { top_candidates: Candidate[], execution_time_ms: number }> {
    return fetchJson(`${API_BASE}/api/agents/workflow`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
    });
}

// --- INGESTION ---

export async function getIngestJob(jobId: string): Promise<{ job_id: string, status: string, source?: string, type?: string }> {
    return fetchJson(`${API_BASE}/api/ingest/jobs/${jobId}`);
}

export async function startIngestion(source: string, payload: Record<string, any>): Promise<{ job_id: string, result?: string }> {
    const endpoint = source === 'all' ? `${API_BASE}/api/ingest/all` : `${API_BASE}/api/ingest/${source}`;
    return fetchJson(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
}

export async function batchIngest(items: any[]): Promise<{ ingested: number }> {
    return fetchJson(`${API_BASE}/api/ingest/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items }),
    });
}
