'use client';

import {
  ExternalLink,
  Loader2,
  RotateCcw,
  Search,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import * as React from 'react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { getEmbeddings, search } from '@/lib/api';
import { SearchResult } from '@/schemas/search';
import { EmbeddingPoint } from '@/schemas/visualization';

function Scatter3DCanvas({
  points,
  selectedPoint,
  onSelectPoint,
  rotation,
  zoom,
}: {
  points: EmbeddingPoint[];
  selectedPoint: EmbeddingPoint | null;
  onSelectPoint: (point: EmbeddingPoint | null) => void;
  rotation: { x: number; y: number };
  zoom: number;
}) {
  const containerRef = React.useRef<HTMLDivElement>(null);

  const getColor = (modality: string) => {
    switch (modality) {
      case 'text':
        return '#3b82f6'; // blue
      case 'molecule':
        return '#22c55e'; // green
      case 'protein':
        return '#f59e0b'; // amber
      default:
        return '#8b5cf6'; // purple
    }
  };

  const project = (point: EmbeddingPoint) => {
    const rad = Math.PI / 180;
    const cosX = Math.cos(rotation.x * rad);
    const sinX = Math.sin(rotation.x * rad);
    const cosY = Math.cos(rotation.y * rad);
    const sinY = Math.sin(rotation.y * rad);

    const x = point.x * cosY - point.z * sinY;
    let z = point.x * sinY + point.z * cosY;

    const y = point.y * cosX - z * sinX;
    z = point.y * sinX + z * cosX;

    const perspective = 500;
    const scale = perspective / (perspective + z * 50);

    return {
      x: 250 + x * 100 * zoom * scale,
      y: 250 - y * 100 * zoom * scale,
      scale,
      z,
    };
  };

  const sortedPoints = [...points]
    .map((p) => ({ ...p, projected: project(p) }))
    .sort((a, b) => a.projected.z - b.projected.z);

  return (
    <div
      ref={containerRef}
      className="relative h-[500px] w-full overflow-hidden rounded-lg bg-linear-to-br from-slate-900 to-slate-800"
      style={{ perspective: '500px' }}
    >
      <svg className="pointer-events-none absolute inset-0 h-full w-full opacity-20">
        <line
          x1="50"
          y1="250"
          x2="450"
          y2="250"
          stroke="#fff"
          strokeWidth="1"
        />
        <line
          x1="250"
          y1="50"
          x2="250"
          y2="450"
          stroke="#fff"
          strokeWidth="1"
        />
        <text x="460" y="255" fill="#fff" fontSize="12">
          X
        </text>
        <text x="255" y="40" fill="#fff" fontSize="12">
          Y
        </text>
      </svg>

      {sortedPoints.map((point) => {
        const { x, y, scale } = point.projected;
        const size = Math.max(6, 12 * scale);
        const isSelected = selectedPoint?.id === point.id;

        return (
          <button
            key={point.id}
            className="absolute cursor-pointer rounded-full transition-all duration-150 hover:ring-2 hover:ring-white/50"
            style={{
              left: x - size / 2,
              top: y - size / 2,
              width: size,
              height: size,
              backgroundColor: getColor(point.modality),
              opacity: 0.5 + scale * 0.5,
              transform: isSelected ? 'scale(1.5)' : 'scale(1)',
              boxShadow: isSelected
                ? `0 0 20px ${getColor(point.modality)}`
                : 'none',
              zIndex: Math.floor(scale * 100),
            }}
            onClick={() => onSelectPoint(isSelected ? null : point)}
            title={point.label}
          />
        );
      })}

      <div className="absolute bottom-4 left-4 flex gap-4 text-xs text-white/70">
        <div className="flex items-center gap-1">
          <div className="h-3 w-3 rounded-full bg-blue-500" />
          <span>Text</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="h-3 w-3 rounded-full bg-green-500" />
          <span>Molecule</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="h-3 w-3 rounded-full bg-amber-500" />
          <span>Protein</span>
        </div>
      </div>
    </div>
  );
}

function EvidencePanel({ result }: { result: SearchResult | null }) {
  if (!result) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="text-lg">Evidence Panel</CardTitle>
          <CardDescription>Select a point to view details</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-muted-foreground py-8 text-center">
            Click on a point in the 3D view to see its evidence trail
          </div>
        </CardContent>
      </Card>
    );
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
          <div className="space-y-4">
            <div>
              <Label className="text-muted-foreground text-xs">Content</Label>
              <p className="bg-muted/50 mt-1 rounded-lg p-3 text-sm">
                {result.content.slice(0, 300)}
                {result.content.length > 300 && '...'}
              </p>
            </div>

            <div>
              <Label className="text-muted-foreground text-xs">Source</Label>
              <Badge className="mt-1">{result.source}</Badge>
            </div>

            {result.citation && (
              <div>
                <Label className="text-muted-foreground text-xs">
                  Citation
                </Label>
                <p className="mt-1 text-sm italic">{result.citation}</p>
              </div>
            )}

            {result.evidence_links && result.evidence_links.length > 0 && (
              <div>
                <Label className="text-muted-foreground text-xs">
                  External Links
                </Label>
                <div className="mt-2 space-y-2">
                  {result.evidence_links.map((link, idx) => (
                    <a
                      key={idx}
                      href={link.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary flex items-center gap-2 text-sm hover:underline"
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
  );
}

function exportToCSV(results: SearchResult[]) {
  const headers = ['id', 'content', 'score', 'modality', 'source', 'citation'];
  const rows = results.map((r) => [
    r.id,
    `"${r.content.replace(/"/g, '""')}"`,
    r.score,
    r.modality,
    r.source,
    r.citation || '',
  ]);

  const csv = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `bioflow_results_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function exportToJSON(results: SearchResult[]) {
  const json = JSON.stringify(results, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `bioflow_results_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function exportToFASTA(results: SearchResult[]) {
  const fasta = results
    .filter((r) => r.modality === 'protein')
    .map((r) => `>${r.id}\n${r.content}`)
    .join('\n\n');

  if (!fasta) {
    alert('No protein sequences to export');
    return;
  }

  const blob = new Blob([fasta], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `bioflow_proteins_${Date.now()}.fasta`;
  a.click();
  URL.revokeObjectURL(url);
}

export default function VisualizationPage() {
  const [isLoading, setIsLoading] = React.useState(false);
  const [query, setQuery] = React.useState('');
  const [points, setPoints] = React.useState<EmbeddingPoint[]>([]);
  const [results, setResults] = React.useState<SearchResult[]>([]);
  const [selectedPoint, setSelectedPoint] =
    React.useState<EmbeddingPoint | null>(null);
  const [selectedResult, setSelectedResult] =
    React.useState<SearchResult | null>(null);
  const [rotation, setRotation] = React.useState({ x: 15, y: 30 });
  const [zoom, setZoom] = React.useState(1);
  const [modalityFilter, setModalityFilter] = React.useState('all');
  const [isDragging, setIsDragging] = React.useState(false);
  const [lastMousePos, setLastMousePos] = React.useState({ x: 0, y: 0 });

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setLastMousePos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - lastMousePos.x;
    const dy = e.clientY - lastMousePos.y;
    setRotation((prev) => ({
      x: prev.x + dy * 0.5,
      y: prev.y + dx * 0.5,
    }));
    setLastMousePos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleSearch = async () => {
    if (!query.trim()) return;

    setIsLoading(true);
    try {
      const data = await search({
        query,
        top_k: 50,
        use_mmr: true,
      });

      if (data.results) {
        let newPoints: EmbeddingPoint[] = [];

        try {
          const embedData = await getEmbeddings(query, 'pca', data.results.length);
          if (embedData.points) {
            newPoints = embedData.points.map((p) => ({
              id: p.id,
              x: p.x,
              y: p.y,
              z: p.z,
              label:
                p.label ||
                (p.content ? String(p.content).slice(0, 50) + '...' : p.id),
              content: p.content || '',
              modality: p.modality || 'text',
              source: p.source || 'unknown',
              score: p.score || 0,
              metadata: p.metadata || {},
            }));
          }
        } catch (e) {
          console.warn('Embedding projection failed:', e);
        }

        if (newPoints.length === 0) {
          throw new Error('No embedding points returned from API');
        }

        setPoints(newPoints);
        setResults(data.results);
      }
    } catch (err) {
      console.error('Search failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const filteredPoints =
    modalityFilter === 'all'
      ? points
      : points.filter((p) => p.modality === modalityFilter);

  const handleSelectPoint = (point: EmbeddingPoint | null) => {
    setSelectedPoint(point);
    if (point) {
      const result = results.find(
        (r) => r.id === point.id || r.content === point.content,
      );
      setSelectedResult(result || null);
    } else {
      setSelectedResult(null);
    }
  };

  return (
    <div className="container mx-auto space-y-6 p-6">
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">
          3D Embedding Explorer
        </h1>
        <p className="text-muted-foreground">
          Visualize and explore multimodal embeddings in 3D space. Search,
          filter, and examine evidence trails.
        </p>
      </div>

      <Card>
        <CardContent className="pt-6">
          <div className="flex gap-4">
            <div className="flex-1">
              <Input
                placeholder="Search for molecules, proteins, or literature..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              />
            </div>
            <Button onClick={handleSearch} disabled={isLoading}>
              {isLoading ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Search className="mr-2 h-4 w-4" />
              )}
              Search
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-4">
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-lg">Controls</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
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

            <div className="space-y-2">
              <Label>Zoom: {zoom.toFixed(1)}x</Label>
              <div className="flex items-center gap-2">
                <ZoomOut className="text-muted-foreground h-4 w-4" />
                <Slider
                  value={[zoom]}
                  onValueChange={([v]) => v !== undefined && setZoom(v)}
                  min={0.5}
                  max={3}
                  step={0.1}
                  className="flex-1"
                />
                <ZoomIn className="text-muted-foreground h-4 w-4" />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Rotation X: {rotation.x.toFixed(0)}°</Label>
              <Slider
                value={[rotation.x]}
                onValueChange={([v]) =>
                  v !== undefined && setRotation((prev) => ({ ...prev, x: v }))
                }
                min={-180}
                max={180}
                step={1}
              />
            </div>

            <div className="space-y-2">
              <Label>Rotation Y: {rotation.y.toFixed(0)}°</Label>
              <Slider
                value={[rotation.y]}
                onValueChange={([v]) =>
                  v !== undefined && setRotation((prev) => ({ ...prev, y: v }))
                }
                min={-180}
                max={180}
                step={1}
              />
            </div>

            <Button
              variant="outline"
              className="w-full"
              onClick={() => {
                setRotation({ x: 15, y: 30 });
                setZoom(1);
              }}
            >
              <RotateCcw className="mr-2 h-4 w-4" />
              Reset View
            </Button>

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

            <div className="border-t pt-4">
              <div className="text-muted-foreground space-y-1 text-sm">
                <p>Points: {filteredPoints.length}</p>
                <p>Results: {results.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>

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
                <div className="text-muted-foreground flex h-[500px] items-center justify-center rounded-lg border border-dashed">
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

        <div className="lg:col-span-1">
          <EvidencePanel result={selectedResult} />
        </div>
      </div>
    </div>
  );
}
