"use client"

import { RefreshCw, SlidersHorizontal } from "lucide-react"
import { useRouter, useSearchParams } from "next/navigation"
import * as React from "react"

import { Button } from "@/components/ui/button"
import { Card, CardContent,CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

export function ExplorerControls() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [isPending, startTransition] = React.useTransition()

  const dataset = searchParams.get("dataset") || "DrugBank"
  const visualization = searchParams.get("view") || "UMAP" 
  const colorBy = searchParams.get("colorBy") || "Activity"

  const createQueryString = React.useCallback(
    (name: string, value: string) => {
      const params = new URLSearchParams(searchParams.toString())
      params.set(name, value)
      return params.toString()
    },
    [searchParams]
  )

  const handleUpdate = (name: string, value: string) => {
    startTransition(() => {
        router.push(`?${createQueryString(name, value)}`, { scroll: false })
    })
  }

  return (
    <Card className="h-full border-l-4 border-l-primary/50">
        <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
                <SlidersHorizontal className="h-5 w-5" />
                Controls
            </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
            <div className="space-y-2">
                <Label htmlFor="dataset">Dataset</Label>
                <Select value={dataset} onValueChange={(v) => handleUpdate("dataset", v)}>
                    <SelectTrigger id="dataset">
                        <SelectValue placeholder="Select dataset" />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectItem value="DrugBank">DrugBank</SelectItem>
                        <SelectItem value="ChEMBL">ChEMBL</SelectItem>
                        <SelectItem value="ZINC">ZINC</SelectItem>
                    </SelectContent>
                </Select>
            </div>
            <div className="space-y-2">
                <Label htmlFor="visualization">Algorithm</Label>
                <Select value={visualization} onValueChange={(v) => handleUpdate("view", v)}>
                    <SelectTrigger id="visualization">
                        <SelectValue placeholder="Select algorithm" />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectItem value="UMAP">UMAP</SelectItem>
                        <SelectItem value="t-SNE">t-SNE</SelectItem>
                        <SelectItem value="PCA">PCA</SelectItem>
                    </SelectContent>
                </Select>
            </div>
            <div className="space-y-2">
                <Label htmlFor="colorBy">Color Mapping</Label>
                <Select value={colorBy} onValueChange={(v) => handleUpdate("colorBy", v)}>
                    <SelectTrigger id="colorBy">
                        <SelectValue placeholder="Select color metric" />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectItem value="Activity">Binding Affinity</SelectItem>
                        <SelectItem value="MW">Molecular Weight</SelectItem>
                        <SelectItem value="LogP">LogP</SelectItem>
                        <SelectItem value="Cluster">Cluster ID</SelectItem>
                    </SelectContent>
                </Select>
            </div>
            
            <div className="pt-4">
                <Button 
                    variant="secondary" 
                    className="w-full" 
                    disabled={isPending}
                    onClick={() => handleUpdate("refresh", Date.now().toString())}
                >
                    <RefreshCw className={`mr-2 h-4 w-4 ${isPending ? "animate-spin" : ""}`} />
                    {isPending ? "Updating..." : "Regenerate View"}
                </Button>
            </div>
        </CardContent>
    </Card>
  )
}
