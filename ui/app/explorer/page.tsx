import { Suspense } from "react"
import { getApiUrl } from "@/lib/api-utils"
import { ExplorerControls } from "./components"
import { ExplorerChart } from "./chart"
import { Loader2 } from "lucide-react"
import { ExplorerResponse } from "@/types/explorer"

interface ExplorerPageProps {
  searchParams?: { [key: string]: string | string[] | undefined }
}

async function getExplorerData(dataset: string, view: string) {
  try {
    const apiUrl = getApiUrl()
    const baseUrl = apiUrl || "http://localhost:3000"

    const res = await fetch(
      `${baseUrl}/api/explorer?dataset=${encodeURIComponent(dataset)}&view=${encodeURIComponent(view)}`,
      { cache: "no-store" }
    )

    if (!res.ok) {
      console.error(`Failed to fetch data: ${res.status} ${res.statusText}`)
      return []
    }

    const json = (await res.json()) as ExplorerResponse
    return json.points || []
  } catch (error) {
    console.error("Error fetching explorer data:", error)
    return []
  }
}

function pickParam(v: string | string[] | undefined) {
  return Array.isArray(v) ? v[0] : v
}

export default async function ExplorerPage({ searchParams }: ExplorerPageProps) {
  const dataset = pickParam(searchParams?.dataset) || "DAVIS"
  const view = pickParam(searchParams?.view) || "PCA"

  const data = await getExplorerData(dataset, view)

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
            key={`${dataset}-${view}`}
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
