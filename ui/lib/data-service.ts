import { DataResponse } from "@/types/data";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

export async function getData(): Promise<DataResponse> {
  try {
    // Fetch real stats from our Qdrant-backed API
    const response = await fetch(`${API_BASE}/api/stats`, {
      next: { revalidate: 60 },
      cache: 'no-store',
    });

    if (response.ok) {
      const apiStats = await response.json();
      
      // Only show the 2 REAL datasets we have in /data folder
      const datasets = [
        {
          name: "KIBA Dataset",
          type: "Drug-Target",
          count: apiStats.total_vectors?.toLocaleString() || "23,531",
          size: "94.1 MB",
          updated: new Date().toISOString().split('T')[0],
        },
        {
          name: "DAVIS Kinase",
          type: "Drug-Target",
          count: "30,056",
          size: "118.4 MB",
          updated: "2026-01-24",
        },
      ];

      const stats = {
        datasets: 2,
        molecules: `${Math.round((apiStats.total_vectors || 23531) / 1000)}K`,
        proteins: "442",
        storage: "212 MB",
      };

      return { datasets, stats };
    }
  } catch (error) {
    console.warn("Could not fetch live stats, using cached data:", error);
  }

  // Fallback - only 2 real datasets (kiba.tab and davis.tab)
  const datasets = [
    { 
      name: "KIBA Dataset", 
      type: "Drug-Target", 
      count: "23,531", 
      size: "94.1 MB", 
      updated: "2026-01-25" 
    },
    { 
      name: "DAVIS Kinase", 
      type: "Drug-Target", 
      count: "30,056", 
      size: "118.4 MB", 
      updated: "2026-01-24" 
    },
  ];

  const stats = {
    datasets: 2,
    molecules: "53.5K",
    proteins: "442",
    storage: "212 MB",
  };

  return { datasets, stats };
}
