import { z } from 'zod';

export const PredictionResultSchema = z.object({
    binding_affinity: z.number(),
    confidence: z.number(),
    interaction_probability: z.number(),
});

export const PredictionResponseSchema = z.object({
    prediction: PredictionResultSchema,
});

export type PredictionResult = z.infer<typeof PredictionResultSchema>;
export type PredictionResponse = z.infer<typeof PredictionResponseSchema>;
