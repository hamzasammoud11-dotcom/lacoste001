"use client"

import { ExternalLink, Loader2, RotateCcw, Search, ZoomIn, ZoomOut } from "lucide-react"
import * as React from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"

// Types
interface EmbeddingPoint {
  id: string
  x: number
  y: number
  z: number
  label: string
  content: string
  modality: string
  source: string
  score?: number
  metadata?: Record<string, unknown>
}

interface SearchResult {
  id: string
  content: string
  score: number
  modality: string
  source: string
  evidence_links?: Array<{
    source: string
    identifier: string
    url: string
    label: string
  }>
  citation?: string
}

// 3D Canvas Component using CSS transforms (no Three.js dependency)
function Scatter3DCanvas({
  points,
  selectedPoint,
  onSelectPoint,
  rotation,
  zoom,
}: {
  points: EmbeddingPoint[]
  selectedPoint: EmbeddingPoint | null
  onSelectPoint: (point: EmbeddingPoint | null) => void
  rotation: { x: number; y: number }
  zoom: number
}) {
  const containerRef = React.useRef<HTMLDivElement>(null)

  // Color by modality
  const getColor = (modality: string) => {
    switch (modality) {
      case "text": return "#3b82f6" // blue
      case "molecule": return "#22c55e" // green
      case "protein": return "#f59e0b" // amber
      default: return "#8b5cf6" // purple
    }
  }

  // Project 3D to 2D with rotation
  const project = (point: EmbeddingPoint) => {
    const rad = Math.PI / 180
    const cosX = Math.cos(rotation.x * rad)
    const sinX = Math.sin(rotation.x * rad)
    const cosY = Math.cos(rotation.y * rad)
    const sinY = Math.sin(rotation.y * rad)

    // Rotate around Y axis
    const x = point.x * cosY - point.z * sinY
    let z = point.x * sinY + point.z * cosY

    // Rotate around X axis
    const y = point.y * cosX - z * sinX
    z = point.y * sinX + z * cosX

    // Simple perspective projection
    const perspective = 500
    const scale = perspective / (perspective + z * 50)
    
    return {
      x: 250 + x * 100 * zoom * scale,
      y: 250 - y * 100 * zoom * scale,
      scale,
      z,
    }
  }

  // Sort points by z for proper rendering order
  const sortedPoints = [...points]
    .map(p => ({ ...p, projected: project(p) }))
    .sort((a, b) => a.projected.z - b.projected.z)

  return (
    <div
      ref={containerRef}
      className="relative w-full h-[500px] bg-gradient-to-br from-slate-900 to-slate-800 rounded-lg overflow-hidden"
      style={{ perspective: "500px" }}
    >
      {/* Axis lines */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-20">
        <line x1="50" y1="250" x2="450" y2="250" stroke="#fff" strokeWidth="1" />
        <line x1="250" y1="50" x2="250" y2="450" stroke="#fff" strokeWidth="1" />
        <text x="460" y="255" fill="#fff" fontSize="12">X</text>
        <text x="255" y="40" fill="#fff" fontSize="12">Y</text>
      </svg>

      {/* Points */}
      {sortedPoints.map((point) => {
        const { x, y, scale } = point.projected
        const size = Math.max(6, 12 * scale)
        const isSelected = selectedPoint?.id === point.id
        
        return (
          <button
            key={point.id}
            className="absolute rounded-full transition-all duration-150 cursor-pointer hover:ring-2 hover:ring-white/50"
            style={{
              left: x - size / 2,
              top: y - size / 2,
              width: size,
              height: size,
              backgroundColor: getColor(point.modality),
              opacity: 0.5 + scale * 0.5,
              transform: isSelected ? "scale(1.5)" : "scale(1)",
              boxShadow: isSelected ? `0 0 20px ${getColor(point.modality)}` : "none",
              zIndex: Math.floor(scale * 100),
            }}
            onClick={() => onSelectPoint(isSelected ? null : point)}
            title={point.label}
          />
        )
      })}

      {/* Legend */}
      <div className="absolute bottom-4 left-4 flex gap-4 text-xs text-white/70">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-blue-500" />
          <span>Text</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span>Molecule</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-amber-500" />
          <span>Protein</span>
        </div>
      </div>
    </div>
  )
}

// Evidence Panel Component
function EvidencePanel({ result }: { result: SearchResult | null }) {
  if (!result) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="text-lg">Evidence Panel</CardTitle>
          <CardDescription>Select a point to view details</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground py-8">
            Click on a point in the 3D view to see its evidence trail
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Evidence</CardTitle>
          <Badge variant="outline">{result.modality}</Badge>
        </div>
        <CardDescription>Score: {result.score.toFixed(3)}</CardDescription>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[400px] pr-4">
          {/* Content */}
          <div className="space-y-4">
            <div>
              <Label className="text-xs text-muted-foreground">Content</Label>
              <p className="text-sm mt-1 bg-muted/50 p-3 rounded-lg">
                {result.content.slice(0, 300)}
                {result.content.length > 300 && "..."}
              </p>
            </div>

            {/* Source */}
            <div>
              <Label className="text-xs text-muted-foreground">Source</Label>
              <Badge className="mt-1">{result.source}</Badge>
            </div>

            {/* Citation */}
            {result.citation && (
              <div>
                <Label className="text-xs text-muted-foreground">Citation</Label>
                <p className="text-sm mt-1 italic">{result.citation}</p>
              </div>
            )}

            {/* Evidence Links */}
            {result.evidence_links && result.evidence_links.length > 0 && (
              <div>
                <Label className="text-xs text-muted-foreground">External Links</Label>
                <div className="mt-2 space-y-2">
                  {result.evidence_links.map((link, idx) => (
                    <a
                      key={idx}
                      href={link.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-sm text-primary hover:underline"
                    >
                      <ExternalLink className="h-3 w-3" />
                      {link.label}
                    </a>
                  ))}
                </div>
              </div>
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}

// Export functions
function exportToCSV(results: SearchResult[]) {
  const headers = ["id", "content", "score", "modality", "source", "citation"]
  const rows = results.map(r => [
    r.id,
    `"${r.content.replace(/"/g, '""')}"`,
    r.score,
    r.modality,
    r.source,
    r.citation || "",
  ])
  
  const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n")
  const blob = new Blob([csv], { type: "text/csv" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = `bioflow_results_${Date.now()}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

function exportToJSON(results: SearchResult[]) {
  const json = JSON.stringify(results, null, 2)
  const blob = new Blob([json], { type: "application/json" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = `bioflow_results_${Date.now()}.json`
  a.click()
  URL.revokeObjectURL(url)
}

function exportToFASTA(results: SearchResult[]) {
  const fasta = results
    .filter(r => r.modality === "protein")
    .map(r => `>${r.id}\n${r.content}`)
    .join("\n\n")
  
  if (!fasta) {
    alert("No protein sequences to export")
    return
  }
  
  const blob = new Blob([fasta], { type: "text/plain" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = `bioflow_proteins_${Date.now()}.fasta`
  a.click()
  URL.revokeObjectURL(url)
}

// Main Visualization Page
export default function VisualizationPage() {
  const [isLoading, setIsLoading] = React.useState(false)
  const [query, setQuery] = React.useState("")
  const [points, setPoints] = React.useState<EmbeddingPoint[]>([])
  const [results, setResults] = React.useState<SearchResult[]>([])
  const [selectedPoint, setSelectedPoint] = React.useState<EmbeddingPoint | null>(null)
  const [selectedResult, setSelectedResult] = React.useState<SearchResult | null>(null)
  const [rotation, setRotation] = React.useState({ x: 15, y: 30 })
  const [zoom, setZoom] = React.useState(1)
  const [modalityFilter, setModalityFilter] = React.useState("all")
  const [isDragging, setIsDragging] = React.useState(false)
  const [lastMousePos, setLastMousePos] = React.useState({ x: 0, y: 0 })

  // Handle mouse drag for rotation
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true)
    setLastMousePos({ x: e.clientX, y: e.clientY })
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return
    const dx = e.clientX - lastMousePos.x
    const dy = e.clientY - lastMousePos.y
    setRotation(prev => ({
      x: prev.x + dy * 0.5,
      y: prev.y + dx * 0.5,
    }))
    setLastMousePos({ x: e.clientX, y: e.clientY })
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  // Search and load embeddings
  const handleSearch = async () => {
    if (!query.trim()) return
    
    setIsLoading(true)
    try {
      const response = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          top_k: 50,
          use_mmr: true,
        }),
      })
      
      const data = await response.json()
      
      if (data.results) {
        let newPoints: EmbeddingPoint[] = []

        try {
          const embedResp = await fetch(
            `/api/explorer/embeddings?query=${encodeURIComponent(query)}&method=pca&limit=${data.results.length}`
          )
          const embedData = await embedResp.json()
          if (embedResp.ok && embedData.points) {
            newPoints = embedData.points.map((p: any) => ({
              id: p.id,
              x: p.x,
              y: p.y,
              z: p.z,
              label: p.label || (p.content ? String(p.content).slice(0, 50) + "..." : p.id),
              content: p.content || "",
              modality: p.modality || "text",
              source: p.source || "unknown",
              score: p.score || 0,
              metadata: p.metadata || {},
            }))
          }
        } catch (e) {
          console.warn("Embedding projection fetch failed, falling back:", e)
        }

        if (newPoints.length === 0) {
          // Fallback to pseudo-positions if embedding API unavailable
          newPoints = data.results.map((r: SearchResult, idx: number) => {
            const angle = (idx / data.results.length) * Math.PI * 2
            const radius = 1 - r.score
            return {
              id: r.id || `point-${idx}`,
              x: Math.cos(angle) * radius + (Math.random() - 0.5) * 0.3,
              y: r.score * 2 - 1 + (Math.random() - 0.5) * 0.2,
              z: Math.sin(angle) * radius + (Math.random() - 0.5) * 0.3,
              label: r.content.slice(0, 50) + "...",
              content: r.content,
              modality: r.modality,
              source: r.source,
              score: r.score,
            }
          })
        }

        setPoints(newPoints)
        setResults(data.results)
      }
    } catch (err) {
      console.error("Search failed:", err)
    } finally {
      setIsLoading(false)
    }
  }

  // Filter points by modality
  const filteredPoints = modalityFilter === "all" 
    ? points 
    : points.filter(p => p.modality === modalityFilter)

  // Handle point selection
  const handleSelectPoint = (point: EmbeddingPoint | null) => {
    setSelectedPoint(point)
    if (point) {
      const result = results.find(r => r.id === point.id || r.content === point.content)
      setSelectedResult(result || null)
    } else {
      setSelectedResult(null)
    }
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">3D Embedding Explorer</h1>
        <p className="text-muted-foreground">
          Visualize and explore multimodal embeddings in 3D space. Search, filter, and examine evidence trails.
        </p>
      </div>

      {/* Search Bar */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex gap-4">
            <div className="flex-1">
              <Input
                placeholder="Search for molecules, proteins, or literature..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              />
            </div>
            <Button onClick={handleSearch} disabled={isLoading}>
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <Search className="h-4 w-4 mr-2" />
              )}
              Search
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Controls */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-lg">Controls</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Modality Filter */}
            <div className="space-y-2">
              <Label>Filter by Modality</Label>
              <Select value={modalityFilter} onValueChange={setModalityFilter}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="text">Text</SelectItem>
                  <SelectItem value="molecule">Molecule</SelectItem>
                  <SelectItem value="protein">Protein</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Zoom */}
            <div className="space-y-2">
              <Label>Zoom: {zoom.toFixed(1)}x</Label>
              <div className="flex items-center gap-2">
                <ZoomOut className="h-4 w-4 text-muted-foreground" />
                <Slider
                  value={[zoom]}
                  onValueChange={([v]) => setZoom(v)}
                  min={0.5}
                  max={3}
                  step={0.1}
                  className="flex-1"
                />
                <ZoomIn className="h-4 w-4 text-muted-foreground" />
              </div>
            </div>

            {/* Rotation controls */}
            <div className="space-y-2">
              <Label>Rotation X: {rotation.x.toFixed(0)}°</Label>
              <Slider
                value={[rotation.x]}
                onValueChange={([v]) => setRotation(prev => ({ ...prev, x: v }))}
                min={-180}
                max={180}
                step={1}
              />
            </div>

            <div className="space-y-2">
              <Label>Rotation Y: {rotation.y.toFixed(0)}°</Label>
              <Slider
                value={[rotation.y]}
                onValueChange={([v]) => setRotation(prev => ({ ...prev, y: v }))}
                min={-180}
                max={180}
                step={1}
              />
            </div>

            {/* Reset */}
            <Button
              variant="outline"
              className="w-full"
              onClick={() => {
                setRotation({ x: 15, y: 30 })
                setZoom(1)
              }}
            >
              <RotateCcw className="h-4 w-4 mr-2" />
              Reset View
            </Button>

            {/* Export */}
            <div className="space-y-2">
              <Label>Export Results</Label>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => exportToCSV(results)}
                  disabled={results.length === 0}
                >
                  CSV
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => exportToJSON(results)}
                  disabled={results.length === 0}
                >
                  JSON
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => exportToFASTA(results)}
                  disabled={results.length === 0}
                >
                  FASTA
                </Button>
              </div>
            </div>

            {/* Stats */}
            <div className="pt-4 border-t">
              <div className="text-sm text-muted-foreground space-y-1">
                <p>Points: {filteredPoints.length}</p>
                <p>Results: {results.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 3D View */}
        <div
          className="lg:col-span-2"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Embedding Space</CardTitle>
              <CardDescription>
                Drag to rotate • Click points for details
              </CardDescription>
            </CardHeader>
            <CardContent className="p-4">
              {filteredPoints.length === 0 && !isLoading ? (
                <div className="h-[500px] flex items-center justify-center text-muted-foreground border border-dashed rounded-lg">
                  No points to display. Run a search to populate embeddings.
                </div>
              ) : (
                <Scatter3DCanvas
                  points={filteredPoints}
                  selectedPoint={selectedPoint}
                  onSelectPoint={handleSelectPoint}
                  rotation={rotation}
                  zoom={zoom}
                />
              )}
            </CardContent>
          </Card>
        </div>

        {/* Evidence Panel */}
        <div className="lg:col-span-1">
          <EvidencePanel result={selectedResult} />
        </div>
      </div>
    </div>
  )
}
