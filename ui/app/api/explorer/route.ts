import { NextResponse } from "next/server";
import { ExplorerResponse, DataPoint } from "@/types/explorer";
import { ExplorerRequestSchema } from "@/schemas/explorer";

const clusters = [
  { cx: 2, cy: 3, color: "var(--color-chart-1)" },
  { cx: -2, cy: -1, color: "var(--color-chart-2)" },
  { cx: 4, cy: -2, color: "var(--color-chart-3)" },
  { cx: -1, cy: 4, color: "var(--color-chart-4)" },
];

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  
  // Validate query params
  const result = ExplorerRequestSchema.safeParse({
      dataset: searchParams.get("dataset") || undefined,
      view: searchParams.get("view") || undefined,
      colorBy: searchParams.get("colorBy") || undefined,
  });

  if (!result.success) {
      return NextResponse.json({ error: result.error }, { status: 400 });
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

  const response: ExplorerResponse = { points, metrics };
  return NextResponse.json(response);
}
