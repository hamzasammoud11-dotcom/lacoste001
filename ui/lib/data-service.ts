import { DataResponse } from "@/types/data";

export async function getData(): Promise<DataResponse> {
  // Mock data simulation - replacing actual database/API call
  const datasets = [
    { name: "DrugBank Compounds", type: "Molecules", count: "12,450", size: "45.2 MB", updated: "2024-01-15" },
    { name: "ChEMBL Kinase Inhibitors", type: "Molecules", count: "8,234", size: "32.1 MB", updated: "2024-01-10" },
    { name: "Custom Protein Targets", type: "Proteins", count: "1,245", size: "78.5 MB", updated: "2024-01-08" },
  ];

  const stats = {
    datasets: 5,
    molecules: "24.5K",
    proteins: "1.2K",
    storage: "156 MB",
  };

  return Promise.resolve({ datasets, stats });
}
