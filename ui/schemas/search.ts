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
