import { z } from 'zod';

export const SearchResultSchema = z.object({
    id: z.string(),
    score: z.number(),
    mmr_score: z.number().optional(),
    content: z.string(),
    modality: z.string(),
    source: z.string().optional(),
    metadata: z.object({
        name: z.string().optional(),
        smiles: z.string().optional(),
        description: z.string().optional(),
        source: z.string().optional(),
        label_true: z.number().optional(),
        affinity_class: z.string().optional(),
        // Image fields
        image: z.string().optional(),
        thumbnail_url: z.string().optional(),
        url: z.string().optional(),
        caption: z.string().optional(),
        image_type: z.string().optional(),
        // Experiment fields (Use Case 4)
        title: z.string().optional(),
        experiment_id: z.string().optional(),
        experiment_type: z.string().optional(),
        outcome: z.string().optional(),
        quality_score: z.number().optional(),
        measurements: z.array(z.object({
            name: z.string(),
            value: z.number(),
            unit: z.string(),
        })).optional(),
        conditions: z.record(z.unknown()).optional(),
        target: z.string().optional(),
        molecule: z.string().optional(),
        protocol: z.string().optional(),
    }).catchall(z.unknown()).optional(),
    citation: z.string().optional(),
    evidence_links: z.array(z.object({
        source: z.string(),
        identifier: z.string(),
        url: z.string(),
        label: z.string(),
    })).optional(),
});

export type SearchResult = z.infer<typeof SearchResultSchema>;
