import { Loader2 } from "lucide-react"
import { Suspense } from "react"

import { getExplorerPoints } from "@/lib/explorer-service"

import { ExplorerChart } from "./chart"
import { ExplorerControls } from "./components"

export const dynamic = 'force-dynamic';

interface ExplorerPageProps {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>
}

function pickParam(v: string | string[] | undefined): string | undefined {
  if (Array.isArray(v)) return v[0]
  return v
}

export default async function ExplorerPage({ searchParams }: ExplorerPageProps) {
  // Await searchParams as required by Next.js 16/15
  const params = await searchParams
  
  const dataset = pickParam(params.dataset) || "DrugBank"
  const view = pickParam(params.view) || "UMAP"
  const colorBy = pickParam(params.colorBy) || "Activity"

  const { points: data } = await getExplorerPoints(dataset, view, colorBy)

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Data Explorer</h1>
        <p className="text-muted-foreground">
          Visualize binding affinity landscapes and model predictions in 3D space using dimensionality reduction.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1">
          <Suspense fallback={<div>Loading controls...</div>}>
            <ExplorerControls />
          </Suspense>
        </div>

        <div className="lg:col-span-3">
          <Suspense
            key={`${dataset}-${view}-${colorBy}`}
            fallback={
              <div className="h-[500px] flex items-center justify-center border rounded-lg bg-muted/10">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            }
          >
            <ExplorerChart data={data} />
          </Suspense>
        </div>
      </div>
    </div>
  )
}
