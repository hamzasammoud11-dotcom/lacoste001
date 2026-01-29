import { z } from 'zod';

export const EmbeddingPointSchema = z.object({
    id: z.string(),
    x: z.number(),
    y: z.number(),
    z: z.number(),
    label: z.string(),
    content: z.string(),
    modality: z.string(),
    source: z.string(),
    score: z.number().optional(),
    metadata: z.record(z.string(), z.unknown()).optional(),
});

export type EmbeddingPoint = z.infer<typeof EmbeddingPointSchema>;
