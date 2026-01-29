import { z } from 'zod';

export const WorkflowStepSchema = z.object({
    id: z.string(),
    type: z.enum(['generate', 'validate', 'rank']),
    name: z.string(),
    config: z.record(z.string(), z.unknown()),
    status: z.enum(['pending', 'running', 'completed', 'error']),
    result: z.unknown().optional(),
    error: z.string().optional(),
});

export const CandidateSchema = z.object({
    smiles: z.string(),
    name: z.string(),
    validation: z.object({
        is_valid: z.boolean(),
        checks: z.record(z.string(), z.boolean()),
        properties: z.record(z.string(), z.number()),
    }),
    score: z.number(),
    rank: z.number().optional(),
});

export const WorkflowResultSchema = z.object({
    candidates: z.array(CandidateSchema),
    steps_completed: z.number(),
    total_time_ms: z.number(),
});

export type Candidate = z.infer<typeof CandidateSchema>;
export type WorkflowStep = z.infer<typeof WorkflowStepSchema>;
export type WorkflowResult = z.infer<typeof WorkflowResultSchema>;
