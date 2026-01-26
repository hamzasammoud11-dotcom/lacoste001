import { ExplorerResponse, DataPoint } from "@/types/explorer";
import { ExplorerRequestSchema } from "@/schemas/explorer";

const clusters = [
  { cx: 2, cy: 3, color: "var(--color-chart-1)" },
  { cx: -2, cy: -1, color: "var(--color-chart-2)" },
  { cx: 4, cy: -2, color: "var(--color-chart-3)" },
  { cx: -1, cy: 4, color: "var(--color-chart-4)" },
];

export async function getExplorerPoints(
  dataset?: string,
  view?: string,
  colorBy?: string
): Promise<ExplorerResponse> {
  // Validate params using the schema to ensure defaults are applied if needed, even for internal calls
  // Note: safeParse is synchronous, but we can treat this as an async service
  const result = ExplorerRequestSchema.safeParse({
    dataset,
    view,
    colorBy,
  });

  if (!result.success) {
    // Return empty/default response or throw type-safe error
    // For now, adhering to existing behavior, we can just throw or fallback
    throw new Error("Invalid parameters");
  }


  // TODO: replace mock generation with backend embeddings
  const points: DataPoint[] = [];
  for (let i = 0; i < 200; i++) {
    const cluster = clusters[Math.floor(i / 50)];
    points.push({
      x: cluster.cx + (Math.random() - 0.5) * 2,
      y: cluster.cy + (Math.random() - 0.5) * 2,
      z: Math.random() * 100,
      color: cluster.color,
      name: `Mol_${i}`,
      affinity: Math.random() * 100,
    });
  }

  const metrics = {
    activeMolecules: 12450,
    clusters: 4,
    avgConfidence: 0.89,
  };

  return Promise.resolve({ points, metrics });
}
