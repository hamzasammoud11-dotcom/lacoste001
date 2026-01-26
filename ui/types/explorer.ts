import { z } from "zod";
import { DataPointSchema, ExplorerMetricsSchema, ExplorerResponseSchema, ExplorerRequestSchema } from "@/schemas/explorer";

export type DataPoint = z.infer<typeof DataPointSchema>;
export type ExplorerMetrics = z.infer<typeof ExplorerMetricsSchema>;
export type ExplorerResponse = z.infer<typeof ExplorerResponseSchema>;
export type ExplorerRequest = z.infer<typeof ExplorerRequestSchema>;
