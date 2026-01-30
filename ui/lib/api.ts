import { API_CONFIG } from '@/config/api.config';
import { DataResponse } from '@/schemas/data';
import { ExplorerRequestSchema, ExplorerResponse } from '@/schemas/explorer';
import { PredictionResponse } from '@/schemas/prediction';
import { SearchResult } from '@/schemas/search';
import { EmbeddingPoint } from '@/schemas/visualization';
import { Candidate,WorkflowResult } from '@/schemas/workflow';
import { Molecule, Protein } from '@/types/visualization';
import { getMockExplorerResponse, generateMock3DEmbeddingPoints, generateMockSearchResults } from './mock-data';

const API_BASE = API_CONFIG.baseUrl;

// --- UTILS ---

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
    const response = await fetch(url, options);
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        // For 400 errors, include the full error details for proper handling
        if (response.status === 400) {
            const errorMessage = error.detail?.message || error.detail || error.error || `HTTP ${response.status}`;
            const err = new Error(errorMessage);
            (err as any).status = 400;
            (err as any).code = error.detail?.error || error.error || 'BAD_REQUEST';
            (err as any).details = error.detail;
            throw err;
        }
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
 * Falls back to mock data if API is unavailable.
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

    try {
        return await fetchJson<ExplorerResponse>(
            `${API_BASE}/api/points?limit=500&view=${apiView}`,
            {
                next: { revalidate: 0 },
                signal: AbortSignal.timeout(5000),
            },
        );
    } catch (error) {
        console.warn('Explorer API unavailable, using mock data:', error);
        return getMockExplorerResponse();
    }
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
 * Query validation result from SMILES/protein detection.
 */
export interface QueryValidation {
    detected_type: 'smiles' | 'protein' | 'text' | 'noise' | 'invalid_smiles' | 'error';
    is_valid_smiles: boolean;
    is_protein_like: boolean;
}

/**
 * Search response with optional query validation and warnings.
 */
export interface SearchResponse {
    results: SearchResult[];
    query?: string;
    modality?: string;
    total_found?: number;
    returned?: number;
    diversity_score?: number | null;
    query_validation?: QueryValidation;
    warning?: string;
    message?: string;
    detail?: string;
    suggestion?: string;
}

/**
 * General search across modalities.
 * Returns query validation info to warn about invalid queries (e.g., "aaa" is not valid SMILES).
 * Falls back to mock data if API is unavailable.
 */
export async function search(params: {
    query: string;
    type?: string;
    limit?: number;
    dataset?: string;
    top_k?: number;
    use_mmr?: boolean;
    include_images?: boolean;
}): Promise<SearchResponse> {
    try {
        return await fetchJson(`${API_BASE}/api/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
    } catch (error) {
        console.warn('Search API unavailable, using mock data:', error);
        const mockResults = generateMockSearchResults(params.top_k || params.limit || 20);
        return {
            results: mockResults,
            query: params.query,
            total_found: mockResults.length,
            returned: mockResults.length,
        };
    }
}

/**
 * Fetches embeddings for a query.
 * Falls back to mock data if API is unavailable.
 */
export async function getEmbeddings(
    query: string,
    method: string = 'pca',
    limit: number = 50,
): Promise<{ points: EmbeddingPoint[] }> {
    try {
        return await fetchJson(
            `${API_BASE}/api/explorer/embeddings?query=${encodeURIComponent(query)}&method=${method}&limit=${limit}`,
        );
    } catch (error) {
        console.warn('Embeddings API unavailable, using mock data:', error);
        return { points: generateMock3DEmbeddingPoints(limit) };
    }
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
 * Image search response with structure detection info.
 */
export interface ImageSearchResponse {
    results: SearchResult[];
    query?: string;
    modality?: string;
    total_found?: number;
    returned?: number;
    message?: string;
    detail?: string;
    suggestion?: string;
    // OCSR (Optical Chemical Structure Recognition) metadata
    ocsr_attempted?: boolean;
    ocsr_success?: boolean;
    ocsr_method?: string;
    ocsr_confidence?: number;
    extracted_smiles?: string;
    ocsr_message?: string;
    ocsr_details?: Record<string, unknown>;
    search_mode?: 'embedding' | 'smiles';
}

/**
 * Search by image - with OCSR (Optical Chemical Structure Recognition).
 * 
 * WHAT THIS DOES:
 * 1. Attempts to extract SMILES from the image using OCSR
 * 2. If successful, searches by the extracted SMILES (structural search)
 * 3. If OCSR fails, falls back to image embedding search
 * 
 * Returns empty results with detailed message if no structure detected.
 * The message explains WHY (e.g., "3D ball-and-stick models need 2D conversion")
 */
export async function searchByImage(params: {
    image: string; // base64 encoded image
    image_type?: string;
    top_k?: number;
    use_mmr?: boolean;
    lambda_param?: number;
    try_ocsr?: boolean; // Attempt OCSR extraction (default: true)
}): Promise<ImageSearchResponse> {
    return fetchJson(`${API_BASE}/api/search/image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image: params.image,
            image_type: params.image_type || 'other',
            top_k: params.top_k || 20,
            use_mmr: params.use_mmr ?? true,
            lambda_param: params.lambda_param || 0.7,
            try_ocsr: params.try_ocsr ?? true,
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
        // Unstructured data (Jury Requirement D.5: Scientific Traceability)
        notes?: string;      // Lab notes excerpt
        abstract?: string;   // Paper abstract excerpt  
        protocol?: string;   // Experimental protocol
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

// --- EXPERIMENTAL IMAGES GALLERY ---

export interface ExperimentalImage {
    id: string;
    score: number;
    content: string;
    modality: string;
    metadata: {
        image_type: 'gel' | 'western_blot' | 'microscopy' | 'fluorescence' | 'spectra';
        description?: string;
        caption?: string;
        image?: string; // base64
        thumbnail_url?: string;
        url?: string;
        source?: string;
        experiment_id?: string;
        experiment_type?: string;
        outcome?: string;
        target_protein?: string;
        cell_line?: string;
        treatment?: string;
        treatment_target?: string;
        concentration?: string;
        conditions?: Record<string, unknown>;
        magnification?: string;
        microscope?: string;
        protocol?: string;
        notes?: string;
        quality_score?: number;
        experiment_date?: string;
        [key: string]: unknown;
    };
}

/**
 * Fetch experimental images (gels, microscopy) for gallery display
 */
export async function getExperimentalImages(params?: {
    type?: 'gel' | 'microscopy' | 'all';
    limit?: number;
    outcome?: string;
    cell_line?: string;
    treatment?: string;
}): Promise<{ images: ExperimentalImage[]; count: number; type: string }> {
    const queryParams = new URLSearchParams();
    if (params?.type) queryParams.set('type', params.type);
    if (params?.limit) queryParams.set('limit', params.limit.toString());
    if (params?.outcome) queryParams.set('outcome', params.outcome);
    if (params?.cell_line) queryParams.set('cell_line', params.cell_line);
    if (params?.treatment) queryParams.set('treatment', params.treatment);
    
    return fetchJson(`/api/images?${queryParams.toString()}`);
}

/**
 * Result from gel/microscopy similarity search
 */
export interface GelMicroscopySimilarResult {
    id: string;
    experiment_id: string;
    image_type: string;
    similarity: number;
    outcome: string;
    conditions: Record<string, unknown>;
    cell_line: string;
    treatment: string;
    concentration: string;
    target_protein: string;
    notes: string;
    protocol: string;
    experiment_type: string;
    magnification: string;
    quality_score: number | null;
    experiment_date: string;
    image?: string; // base64
}

export interface GelMicroscopySearchResponse {
    results: GelMicroscopySimilarResult[];
    query_image_type: string | null;
    total_found: number;
    returned: number;
    filters_applied: Record<string, unknown>;
    message: string;
}

/**
 * Search for similar biological images (gels, microscopy)
 * 
 * Use Case 4: Upload a Western blot or microscopy image and find
 * experiments with similar visual patterns.
 */
export async function searchGelMicroscopy(params: {
    image: string; // base64 data URL
    image_type?: 'gel' | 'western_blot' | 'microscopy' | 'fluorescence';
    outcome?: string;
    cell_line?: string;
    treatment?: string;
    top_k?: number;
    use_mmr?: boolean;
}): Promise<GelMicroscopySearchResponse> {
    return fetchJson('/api/search/gel-microscopy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
    });
}

/**
 * Get available filter options for the gallery
 */
export interface ImageFilterOptions {
    image_types: string[];
    outcomes: string[];
    cell_lines: string[];
    treatments: string[];
}

export async function getImageFilterOptions(): Promise<ImageFilterOptions> {
    return fetchJson('/api/images/filters');
}


// --- CROSS-MODAL SEARCH ---

export interface CrossModalSearchParams {
    compound?: string;    // SMILES string
    sequence?: string;    // DNA/RNA/protein sequence
    text?: string;        // Text query
    image?: string;       // Base64 encoded image
    target_modalities?: string[];  // molecule, protein, text, image, experiment, all
    top_k?: number;
    use_mmr?: boolean;
    diversity?: number;
}

export interface CrossModalResult {
    id: string;
    score: number;
    content: string;
    modality: string;
    query_source: string;
    source_modality?: string;
    connection: string;
    related_items?: Array<{ id: string; modality: string; score: number }>;
    metadata: {
        source?: string;
        experiment_id?: string;
        image_type?: string;
        image?: string;
        name?: string;
        description?: string;
        target?: string;
        experiment_type?: string;
        [key: string]: unknown;
    };
}

export interface CrossModalSearchResponse {
    results: CrossModalResult[];
    query_info: Record<string, unknown>;
    total_found: number;
    returned: number;
    target_modalities: string[];
    message: string;
    validation_warnings?: string[];
}

/**
 * Cross-modal search: combine compound, sequence, text, or image to find related experiments.
 * 
 * Use Case 4: "Show me experiments that used THIS compound with THIS gel result"
 * 
 * Examples:
 * - Search by compound: { compound: "CCO" }
 * - Search by sequence: { sequence: "MKTAYIAK..." }
 * - Search by text: { text: "EGFR inhibitor" }
 * - Combined: { compound: "CCO", text: "binding assay" }
 */
export async function searchCrossModal(params: CrossModalSearchParams): Promise<CrossModalSearchResponse> {
    return fetchJson('/api/search/cross-modal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            compound: params.compound,
            sequence: params.sequence,
            text: params.text,
            image: params.image,
            target_modalities: params.target_modalities ?? ['all'],
            top_k: params.top_k ?? 10,
            use_mmr: params.use_mmr ?? true,
            diversity: params.diversity ?? 0.3,
        }),
    });
}

// --- EVIDENCE CHAIN & EXPLORE FROM HERE ---

/**
 * Evidence chain node for visualization
 */
export interface EvidenceNode {
    id: string;
    type: 'compound' | 'experiment' | 'paper' | 'protein' | 'image';
    label: string;
    subtitle?: string;
    score?: number;
    url?: string;
    metadata?: Record<string, unknown>;
}

/**
 * Evidence chain edge connecting nodes
 */
export interface EvidenceEdge {
    from: string;
    to: string;
    relationship: string;
    strength?: number;
}

/**
 * Complete evidence chain for visualization
 */
export interface EvidenceChain {
    nodes: EvidenceNode[];
    edges: EvidenceEdge[];
    rootId: string;
}

/**
 * Get evidence chain for an item showing connected relationships.
 * 
 * Returns a graph showing:
 * Compound X → tested in Experiment Y → similar to Experiment Z → cited in Paper ABC
 */
export async function getEvidenceChain(params: {
    item_id: string;
    item_type?: string;
    depth?: number;
    include_papers?: boolean;
}): Promise<EvidenceChain & { message: string }> {
    return fetchJson(`${API_BASE}/api/evidence-chain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            item_id: params.item_id,
            item_type: params.item_type,
            depth: params.depth ?? 3,
            include_papers: params.include_papers ?? true,
        }),
    });
}

/**
 * Related item for explore from here navigation
 */
export interface RelatedItem {
    id: string;
    type: 'compound' | 'experiment' | 'paper' | 'protein' | 'image' | 'sequence';
    title: string;
    subtitle?: string;
    score?: number;
    url?: string;
    metadata?: Record<string, unknown>;
}

/**
 * Category of related items
 */
export interface ExploreCategory {
    id: string;
    label: string;
    icon: string;
    count: number;
    items: RelatedItem[];
    loading?: boolean;
}

/**
 * Response from explore from here API
 */
export interface ExploreFromHereResponse {
    sourceId: string;
    sourceType: string;
    sourceTitle: string;
    categories: ExploreCategory[];
    message: string;
}

/**
 * Get related items for "Explore from Here" navigation.
 * 
 * Click any result → jump to related:
 * - Compounds used in this experiment
 * - Similar experiments
 * - Papers citing this
 * - Sequences targeted
 */
export async function exploreFromHere(
    itemId: string,
    itemType: string = 'auto'
): Promise<ExploreFromHereResponse> {
    return fetchJson(`${API_BASE}/api/explore/${encodeURIComponent(itemId)}?type=${encodeURIComponent(itemType)}`);
}

// --- ENHANCED FACETED FILTERING ---

/**
 * Filter condition for advanced filtering
 */
export interface FilterCondition {
    field: string;
    operator: '=' | '!=' | '>' | '<' | '>=' | '<=' | 'in' | 'contains';
    values: string[];
    numericValue?: number;
}

/**
 * Filter group with AND/OR operator
 */
export interface FilterGroup {
    operator: 'AND' | 'OR';
    conditions: FilterCondition[];
}

/**
 * Complete filter configuration
 */
export interface FilterConfig {
    groups: FilterGroup[];
    globalOperator: 'AND' | 'OR';
}

/**
 * Search with enhanced faceted filtering.
 * 
 * Multi-select + boolean logic:
 * [Outcome: positive OR negative] AND [Cell Line: HeLa, U2OS] AND [Quality > 0.8]
 */
export async function searchWithFilters(params: {
    query?: string;
    filters: Record<string, string | string[] | { min?: number; max?: number }>;
    filter_operator?: 'AND' | 'OR';
    top_k?: number;
}): Promise<{
    results: SearchResult[];
    total_found: number;
    filters_applied: Record<string, unknown>;
    filter_operator: string;
    facet_counts: Record<string, Record<string, number>>;
}> {
    return fetchJson(`${API_BASE}/api/search/filtered`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query: params.query,
            filters: params.filters,
            filter_operator: params.filter_operator ?? 'AND',
            top_k: params.top_k ?? 20,
        }),
    });
}

// --- DESIGN VARIANT JUSTIFICATION ---

/**
 * Chemical modification in a variant
 */
export interface ChemicalModification {
    position: string;
    originalGroup: string;
    newGroup: string;
    effect: string;
    confidence: number;
}

/**
 * Similar compound with known results
 */
export interface SimilarCompound {
    id: string;
    name: string;
    smiles?: string;
    experimentId?: string;
    outcome: 'success' | 'partial' | 'failure';
    activity?: string;
    url?: string;
}

/**
 * Predicted property with comparison to parent
 */
export interface PredictedProperty {
    name: string;
    value: number | string;
    unit?: string;
    comparison?: {
        baseline: number | string;
        improvement: number;
        direction: 'better' | 'worse' | 'neutral';
    };
    confidence?: number;
}

/**
 * Evidence source for justification
 */
export interface EvidenceSource {
    type: 'paper' | 'experiment' | 'database' | 'prediction';
    id: string;
    title: string;
    url?: string;
    relevance: number;
}

/**
 * Full variant justification data
 */
export interface VariantJustificationData {
    variantId: string;
    summary: string;
    reasoning: string[];
    modifications: ChemicalModification[];
    similarCompounds: SimilarCompound[];
    predictedProperties: PredictedProperty[];
    evidenceSources: EvidenceSource[];
    overallConfidence: number;
    riskFactors?: string[];
    suggestedExperiments?: string[];
}

/**
 * Get detailed justification for a design variant.
 * 
 * Returns:
 * - Why this variant was suggested
 * - Specific chemical modifications and their effects
 * - Similar compounds that succeeded/failed
 * - Predicted property improvements (e.g., IC50: 45 nM vs parent 250 nM)
 */
export async function getVariantJustification(params: {
    reference: string;
    variant_content: string;
    modality?: string;
}): Promise<VariantJustificationData> {
    // This would typically be a separate API call, but for now we extract from the variant data
    return fetchJson(`${API_BASE}/api/design/justification`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
    });
}

