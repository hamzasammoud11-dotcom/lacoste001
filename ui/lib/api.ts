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
    include_images?: boolean;
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

// --- IMAGE SEARCH (Use Case 4) ---

/**
 * Search by image - encodes image and finds similar items.
 */
export async function searchByImage(params: {
    image: string; // base64 encoded image
    image_type?: string;
    top_k?: number;
    use_mmr?: boolean;
    lambda_param?: number;
}): Promise<{ results: SearchResult[] }> {
    return fetchJson(`${API_BASE}/api/search/image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image: params.image,
            image_type: params.image_type || 'other',
            top_k: params.top_k || 20,
            use_mmr: params.use_mmr ?? true,
            lambda_param: params.lambda_param || 0.7,
        }),
    });
}

// --- NEIGHBOR EXPLORATION (Use Case 4) ---

/**
 * Find neighbors of an item for guided exploration.
 */
export async function searchNeighbors(params: {
    item_id: string;
    collection?: string;
    top_k?: number;
    include_cross_modal?: boolean;
    diversity?: number;
}): Promise<{
    source_id: string;
    source_modality: string;
    neighbors: Array<{
        id: string;
        score: number;
        content: string;
        modality: string;
        collection: string;
        metadata: Record<string, unknown>;
    }>;
    facets: Record<string, number>;
    total_found: number;
}> {
    return fetchJson(`${API_BASE}/api/neighbors`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            item_id: params.item_id,
            collection: params.collection,
            top_k: params.top_k || 20,
            include_cross_modal: params.include_cross_modal ?? true,
            diversity: params.diversity || 0.3,
        }),
    });
}

// --- EXPERIMENT SEARCH (Use Case 4) ---

/**
 * Search experiments with outcome-based filtering.
 */
export async function searchExperiments(params: {
    query: string;
    experiment_type?: string;
    outcome?: string;
    target?: string;
    quality_min?: number;
    top_k?: number;
}): Promise<{
    query: string;
    experiments: Array<{
        id: string;
        score: number;
        experiment_id: string;
        title: string;
        experiment_type: string;
        outcome: string;
        quality_score: number;
        measurements: Array<{ name: string; value: number; unit: string }>;
        conditions: Record<string, unknown>;
        target: string;
        molecule: string;
        description: string;
        evidence_links: Array<{ source: string; identifier: string; url: string }>;
    }>;
    total_found: number;
}> {
    return fetchJson(`${API_BASE}/api/experiments/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
    });
}

// --- DESIGN VARIANTS (Use Case 4) ---

/**
 * Get design variant suggestions with justifications.
 */
export async function getDesignVariants(params: {
    reference: string;
    modality?: string;
    num_variants?: number;
    diversity?: number;
    constraints?: Record<string, unknown>;
}): Promise<{
    reference: string;
    reference_modality: string;
    variants: Array<{
        rank: number;
        id: string;
        content: string;
        modality: string;
        similarity_score: number;
        diversity_score: number;
        justification: string;
        evidence_links: Array<{ source: string; identifier: string; url: string }>;
        metadata: Record<string, unknown>;
    }>;
    num_returned: number;
}> {
    return fetchJson(`${API_BASE}/api/design/variants`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            reference: params.reference,
            modality: params.modality || 'auto',
            num_variants: params.num_variants || 5,
            diversity: params.diversity || 0.5,
            constraints: params.constraints,
        }),
    });
}
