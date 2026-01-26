import { z } from "zod";
import { DatasetSchema, StatisticsSchema, DataResponseSchema } from "@/schemas/data";

export type Dataset = z.infer<typeof DatasetSchema>;
export type Statistics = z.infer<typeof StatisticsSchema>;
export type DataResponse = z.infer<typeof DataResponseSchema>;
