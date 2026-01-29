import { z } from 'zod';

export const DatasetSchema = z.object({
  name: z.string(),
  type: z.string(),
  count: z.string(),
  size: z.string(),
  updated: z.string(),
});

export const StatisticsSchema = z.object({
  datasets: z.number(),
  molecules: z.string(),
  proteins: z.string(),
  storage: z.string(),
});

export const DataResponseSchema = z.object({
  datasets: z.array(DatasetSchema),
  stats: StatisticsSchema,
});

export type Dataset = z.infer<typeof DatasetSchema>;
export type Statistics = z.infer<typeof StatisticsSchema>;
export type DataResponse = z.infer<typeof DataResponseSchema>;
