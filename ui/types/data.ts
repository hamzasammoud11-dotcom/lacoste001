import { z } from "zod";

import { DataResponseSchema,DatasetSchema, StatisticsSchema } from "@/schemas/data";

export type Dataset = z.infer<typeof DatasetSchema>;
export type Statistics = z.infer<typeof StatisticsSchema>;
export type DataResponse = z.infer<typeof DataResponseSchema>;
