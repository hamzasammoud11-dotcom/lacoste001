import { API_CONFIG } from "@/config/api.config";
import { ExplorerRequestSchema } from "@/schemas/explorer";
import { DataPoint, ExplorerResponse } from "@/types/explorer";

export async function getExplorerPoints(
  dataset?: string,
  view?: string,
  colorBy?: string,
  query?: string,
  limit?: number,
  modality?: string
): Promise<ExplorerResponse> {
  const result = ExplorerRequestSchema.safeParse({ dataset, view, colorBy, query, limit, modality });

  if (!result.success) {
    throw new Error("Invalid parameters");
  }

  // Map view param to API view type
  const apiView = view === "UMAP" ? "combined" : view === "PCA-Drug" ? "drug" : view === "PCA-Target" ? "target" : "combined";
  const fetchLimit = limit || 500;

  // Phase 3: Fetch from Python API with pre-computed PCA
  try {
    const res = await fetch(`${API_CONFIG.baseUrl}/api/points?limit=${fetchLimit}&view=${apiView}`, {
        next: { revalidate: 0 },
        signal: AbortSignal.timeout(5000)
    });

    if (res.ok) {
        const data = await res.json();
        
        // Map API response to UI schema - now using real PCA coordinates!
        const realPoints = data.points.map((p: any) => ({
             x: p.x,
             y: p.y,
             z: p.z,
             color: p.color || "#64748b",
             name: p.name || "Unknown",
             affinity: p.affinity || 0,
        }));

        return {
            points: realPoints,
            metrics: data.metrics || {
                activeMolecules: realPoints.length,
                clusters: 3,
                avgConfidence: 0.80
            }
        };
    }
  } catch (e) {
      console.warn("Python backend unreachable, using mock data:", e);
  }

  // --- FALLBACK MOCK DATA ---
  const clusters = [
    { cx: 2, cy: 3, cz: 1, color: "var(--color-chart-1)" },
    { cx: -2, cy: -1, cz: -1, color: "var(--color-chart-2)" },
    { cx: 4, cy: -2, cz: 2, color: "var(--color-chart-3)" },
    { cx: -1, cy: 4, cz: 0, color: "var(--color-chart-4)" },
  ];

  const points = Array.from({ length: 500 }).map((_, i) => {
    const cluster = clusters[Math.floor(Math.random() * clusters.length)];
    const spread = 1.5;
    return {
      x: cluster.cx + (Math.random() - 0.5) * spread,
      y: cluster.cy + (Math.random() - 0.5) * spread,
      z: cluster.cz + (Math.random() - 0.5) * spread,
      color: cluster.color,
      name: `MOL-${1000 + i}`,
      affinity: Math.random(),
    };
  });

  return Promise.resolve({
    points,
    metrics: {
      activeMolecules: 12450,
      clusters: 4,
      avgConfidence: 0.85,
    },
  });
}
