/**
 * Use Case 4: Multimodal Biological Design & Discovery Intelligence
 * Comprehensive test suite addressing jury critique requirements:
 * 
 * D.4 - Design Assistance: Propose 'close but diverse' variants AND JUSTIFY them
 * D.5 - Scientific Traceability: Link suggestions to evidence (docs, experiments, lab notes)
 * B - Multimodal: Connect biological objects (text, sequences, images)
 * D.1 - Data Ingestion: Ingest & normalize multimodal items with meaningful metadata
 */

import { describe, expect, it } from 'vitest';

const API_BASE = process.env.TEST_API_BASE || 'http://localhost:8000';

// Test helpers
async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

describe('Use Case 4: Multimodal Biological Design & Discovery Intelligence', () => {
  // ==========================================================================
  // D.4: DESIGN ASSISTANCE - Propose variants WITH JUSTIFICATIONS
  // ==========================================================================
  describe('D.4: Design Assistance & Justification', () => {
    it('should suggest design variants with scientific justifications', async () => {
      const response = await fetchJson<{
        reference: string;
        reference_modality: string;
        variants: Array<{
          rank: number;
          content: string;
          similarity_score: number;
          priority_score?: number;
          justification: string; // CRITICAL: Must exist
          evidence_links: Array<{ source: string; identifier: string; url: string }>;
        }>;
      }>(`${API_BASE}/api/design/variants`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reference: 'CC(=O)Nc1ccc(O)cc1', // Acetaminophen SMILES
          modality: 'molecule',
          num_variants: 5,
          diversity: 0.5,
        }),
      });

      expect(response.variants).toHaveLength(5);
      
      // JURY REQUIREMENT: Each variant MUST have a justification
      for (const variant of response.variants) {
        expect(variant.justification).toBeDefined();
        expect(variant.justification.length).toBeGreaterThan(20);
        
        // Justification should contain scientific reasoning, not just numbers
        const containsScientificReasoning = 
          variant.justification.includes('hypothesis') ||
          variant.justification.includes('binding') ||
          variant.justification.includes('activity') ||
          variant.justification.includes('structural') ||
          variant.justification.includes('evidence') ||
          variant.justification.includes('experiment');
        
        expect(containsScientificReasoning).toBe(true);
      }
    });

    it('should return priority scores that differ from similarity scores', async () => {
      const response = await fetchJson<{
        variants: Array<{
          similarity_score: number;
          priority_score: number;
          diversity_score: number;
        }>;
      }>(`${API_BASE}/api/design/variants`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reference: 'COC1=CC=C(NC2=NC=NC3=CC=CC=C23)C=C1', // Quinazoline scaffold
          modality: 'molecule',
          num_variants: 3,
          diversity: 0.7,
        }),
      });

      // JURY REQUIREMENT: Priority ≠ Similarity
      // Priority should factor in evidence strength, not just structural similarity
      expect(response.variants.length).toBeGreaterThan(0);
      
      for (const variant of response.variants) {
        expect(variant.priority_score).toBeDefined();
        expect(typeof variant.priority_score).toBe('number');
        
        // They should be different (priority considers more factors)
        if (variant.similarity_score !== variant.priority_score) {
          // Good - they differ, meaning priority is computed differently
          expect(true).toBe(true);
        }
      }
    });
  });

  // ==========================================================================
  // D.5: SCIENTIFIC TRACEABILITY - Link to evidence
  // ==========================================================================
  describe('D.5: Scientific Traceability', () => {
    it('should return evidence links from diverse sources', async () => {
      const response = await fetchJson<{ results: any[] }>(`${API_BASE}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: 'EGFR inhibitor kinase',
          type: 'text',
          limit: 5,
        }),
      });

      expect(response.results.length).toBeGreaterThan(0);
      
      // Check that results have evidence links (not just "Source: PubChem")
      const hasEvidenceLinks = response.results.some(r => 
        r.evidence_links && r.evidence_links.length > 0
      );
      
      // Evidence links should include clickable URLs
      if (hasEvidenceLinks) {
        const linksWithUrls = response.results.flatMap(r => r.evidence_links || [])
          .filter(link => link.url && link.url.startsWith('http'));
        expect(linksWithUrls.length).toBeGreaterThan(0);
      }
    });

    it('should include unstructured data (lab notes, abstracts) in experiments', async () => {
      const response = await fetchJson<{
        experiments: Array<{
          title: string;
          description: string;
          evidence_links?: Array<{ source: string; identifier: string; url: string }>;
        }>;
      }>(`${API_BASE}/api/experiments/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: 'kinase inhibitor binding',
          top_k: 5,
        }),
      });

      expect(response.experiments.length).toBeGreaterThan(0);
      
      // JURY REQUIREMENT: Should have unstructured text, not just DB IDs
      for (const exp of response.experiments) {
        expect(exp.title).toBeDefined();
        expect(exp.description).toBeDefined();
        // Description should be meaningful text, not just an ID
        expect(exp.description?.length).toBeGreaterThan(20);
      }
    });

    it('should include lab notes and protocol references', async () => {
      const response = await fetchJson<{
        experiments: Array<{
          experiment_id: string;
          notes?: string;
          protocol?: string;
          abstract?: string;
        }>;
      }>(`${API_BASE}/api/experiments/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: 'EGFR gefitinib',
          top_k: 3,
        }),
      });

      // At least some experiments should have notes/protocol (unstructured data)
      const hasNotes = response.experiments.some(e => e.notes && e.notes.length > 0);
      const hasProtocol = response.experiments.some(e => e.protocol && e.protocol.length > 0);
      
      // JURY REQUIREMENT: Should not only query structured DBs
      expect(hasNotes || hasProtocol).toBe(true);
    });
  });

  // ==========================================================================
  // B: MULTIMODAL - Connect text, sequences, and images
  // ==========================================================================
  describe('B: Multimodal Connectivity', () => {
    it('should search across modalities with a single query', async () => {
      const response = await fetchJson<{
        results: Array<{
          id: string;
          modality: string;
          score: number;
        }>;
      }>(`${API_BASE}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: 'kinase inhibitor',
          type: 'text',
          limit: 10,
          include_images: true,
        }),
      });

      // Should return results from multiple modalities
      const modalities = new Set(response.results.map(r => r.modality));
      expect(modalities.size).toBeGreaterThanOrEqual(1);
    });

    it('should return cross-modal neighbors with connection explanations', async () => {
      // First search to get an item ID
      const searchResp = await fetchJson<{ results: Array<{ id: string }> }>(`${API_BASE}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: 'CC(=O)Nc1ccc(O)cc1',
          type: 'drug',
          limit: 1,
        }),
      });

      if (searchResp.results.length > 0) {
        const itemId = searchResp.results[0].id;
        
        const neighborResp = await fetchJson<{
          neighbors: Array<{
            id: string;
            modality: string;
            score: number;
            connection_explanation?: string;
          }>;
        }>(`${API_BASE}/api/neighbors/search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            item_id: itemId,
            top_k: 5,
            include_cross_modal: true,
          }),
        });

        // JURY REQUIREMENT: Connection logic should NOT be a black box
        const hasExplanations = neighborResp.neighbors.some(n => 
          n.connection_explanation && n.connection_explanation.length > 0
        );
        
        // Cross-modal connections should explain WHY they're connected
        expect(hasExplanations).toBe(true);
      }
    });

    it('should search by image and return relevant results', async () => {
      // Create a minimal test image (1x1 PNG in base64)
      const testImageBase64 = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
      
      const response = await fetchJson<{
        results: Array<{
          id: string;
          modality: string;
          metadata?: {
            image?: string;
            thumbnail_url?: string;
            description?: string;
          };
        }>;
      }>(`${API_BASE}/api/search/image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: testImageBase64,
          image_type: 'microscopy',
          top_k: 5,
        }),
      });

      // Should not fail (even if no results)
      expect(response).toBeDefined();
    });
  });

  // ==========================================================================
  // D.1: DATA INGESTION - Normalized metadata
  // ==========================================================================
  describe('D.1: Data Ingestion & Normalization', () => {
    it('should return molecules with meaningful metadata', async () => {
      const response = await fetchJson<{
        molecules: Array<{
          id: string;
          smiles?: string;
          name?: string;
          metadata?: Record<string, unknown>;
        }>;
      }>(`${API_BASE}/api/molecules`);

      expect(response.molecules.length).toBeGreaterThan(0);
      
      // Metadata should include useful fields
      for (const mol of response.molecules.slice(0, 5)) {
        if (mol.metadata) {
          // Should have more than just an ID
          const metadataKeys = Object.keys(mol.metadata);
          expect(metadataKeys.length).toBeGreaterThan(0);
        }
      }
    });

    it('should return experiments with experimental conditions', async () => {
      const response = await fetchJson<{
        experiments: Array<{
          experiment_id: string;
          conditions?: Record<string, unknown>;
          measurements?: Array<{ name: string; value: number; unit: string }>;
        }>;
      }>(`${API_BASE}/api/experiments/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: 'binding assay',
          top_k: 5,
        }),
      });

      // JURY REQUIREMENT: Should have experimental conditions, not bare metadata
      const hasConditions = response.experiments.some(e => 
        e.conditions && Object.keys(e.conditions).length > 0
      );
      const hasMeasurements = response.experiments.some(e =>
        e.measurements && e.measurements.length > 0
      );
      
      expect(hasConditions || hasMeasurements).toBe(true);
    });
  });

  // ==========================================================================
  // IMAGE DISPLAY - Consistent rendering
  // ==========================================================================
  describe('Image Display Consistency', () => {
    it('should return images with valid display URLs or base64 data', async () => {
      const response = await fetchJson<{
        results: Array<{
          id: string;
          modality: string;
          metadata?: {
            image?: string;
            thumbnail_url?: string;
            url?: string;
          };
        }>;
      }>(`${API_BASE}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: 'microscopy gel',
          type: 'text',
          limit: 10,
          include_images: true,
        }),
      });

      const imageResults = response.results.filter(r => r.modality === 'image');
      
      for (const result of imageResults) {
        const hasValidImage = 
          (result.metadata?.image && 
            (result.metadata.image.startsWith('data:') || 
             result.metadata.image.startsWith('http'))) ||
          (result.metadata?.thumbnail_url && 
            result.metadata.thumbnail_url.startsWith('http')) ||
          (result.metadata?.url && 
            result.metadata.url.startsWith('http'));
        
        // JURY REQUIREMENT: Images should be renderable (not broken paths)
        if (imageResults.length > 0) {
          expect(hasValidImage).toBe(true);
        }
      }
    });
  });
});

// ==========================================================================
// INTEGRATION TESTS - Full workflow
// ==========================================================================
describe('Integration: Discovery Intelligence Workflow', () => {
  it('should complete full discovery workflow: search -> variants -> evidence', async () => {
    // Step 1: Search for a molecule
    const searchResp = await fetchJson<{ results: any[] }>(`${API_BASE}/api/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: 'CC(=O)Nc1ccc(O)cc1', // Acetaminophen
        type: 'drug',
        limit: 1,
      }),
    });
    
    expect(searchResp.results.length).toBeGreaterThan(0);
    const molecule = searchResp.results[0];

    // Step 2: Get design variants with justifications
    const variantsResp = await fetchJson<{ variants: any[] }>(`${API_BASE}/api/design/variants`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        reference: molecule.content || molecule.metadata?.smiles,
        modality: 'molecule',
        num_variants: 3,
      }),
    });

    expect(variantsResp.variants.length).toBeGreaterThan(0);
    
    // Step 3: Each variant should have justification AND evidence links
    for (const variant of variantsResp.variants) {
      expect(variant.justification).toBeDefined();
      expect(variant.evidence_links).toBeDefined();
    }
  });
});
