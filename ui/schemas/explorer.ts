import { z } from "zod";

export const DataPointSchema = z.object({
  x: z.number(),
  y: z.number(),
  z: z.number(),
  color: z.string(),
  name: z.string(),
  affinity: z.number(),
});

export const ExplorerMetricsSchema = z.object({
  activeMolecules: z.number(),
  clusters: z.number(),
  avgConfidence: z.number(),
});

export const ExplorerResponseSchema = z.object({
  points: z.array(DataPointSchema),
  metrics: ExplorerMetricsSchema,
});

export const ExplorerRequestSchema = z.object({
  dataset: z.string().optional().default("DrugBank"),
  view: z.string().optional().default("UMAP"),
  colorBy: z.string().optional().default("Activity"),
});
