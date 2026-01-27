'use client'

import { Loader2 } from "lucide-react"
import { useState } from "react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"

interface PredictionResult {
  binding_affinity: number
  confidence: number
  interaction_probability: number
}

export function ExplorerPredictions() {
  const [drugSmiles, setDrugSmiles] = useState("")
  const [targetSequence, setTargetSequence] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)

  const handlePredict = async () => {
    setIsLoading(true)
    setError(null)
    setResult(null)
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          drug_smiles: drugSmiles,
          target_sequence: targetSequence,
        }),
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data?.error || data?.detail || "Prediction failed")
      }
      setResult(data.prediction || null)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Predictions</CardTitle>
        <CardDescription>Estimate drug–target interaction from SMILES and protein sequence.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertTitle>Prediction error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="drug-smiles">Drug SMILES</Label>
            <Input
              id="drug-smiles"
              placeholder="CC(=O)OC1=CC=CC=C1C(=O)O"
              value={drugSmiles}
              onChange={(e) => setDrugSmiles(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="target-seq">Protein Sequence</Label>
            <Textarea
              id="target-seq"
              placeholder="MKWVTFISLLFLFSSAYSRGVFRR..."
              className="min-h-[90px] font-mono"
              value={targetSequence}
              onChange={(e) => setTargetSequence(e.target.value)}
            />
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Button onClick={handlePredict} disabled={isLoading || !drugSmiles || !targetSequence}>
            {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
            Run Prediction
          </Button>
          {result && (
            <div className="text-sm text-muted-foreground">
              Affinity: <span className="font-medium text-foreground">{result.binding_affinity.toFixed(4)}</span>
              {" · "}
              Confidence: <span className="font-medium text-foreground">{result.confidence.toFixed(2)}</span>
              {" · "}
              Probability: <span className="font-medium text-foreground">{result.interaction_probability.toFixed(2)}</span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
