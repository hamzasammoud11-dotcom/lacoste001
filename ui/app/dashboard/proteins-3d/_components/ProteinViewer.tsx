'use client';

import { AlertCircle, RotateCcw } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { getProteinPdbUrl } from '@/lib/visualization-api';
import type { ProteinRepresentation } from '@/lib/visualization-types';

interface ProteinViewerProps {
  pdbId: string;
  width?: number;
  height?: number;
  className?: string;
  initialRepresentation?: ProteinRepresentation;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type $3Dmol = any;

export function ProteinViewer({
  pdbId,
  width = 500,
  height = 500,
  className,
  initialRepresentation = 'cartoon',
}: ProteinViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const viewerRef = useRef<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [representation, setRepresentation] =
    useState<ProteinRepresentation>(initialRepresentation);
  const [$3Dmol, set$3Dmol] = useState<$3Dmol | null>(null);
  const [pdbData, setPdbData] = useState<string | null>(null);

  // Load 3Dmol.js dynamically
  useEffect(() => {
    const load3Dmol = async () => {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const module = await import('3dmol') as any;
        set$3Dmol(module);
      } catch (err) {
        console.error('Failed to load 3Dmol:', err);
        setError('Failed to load 3D visualization library');
        setIsLoading(false);
      }
    };
    load3Dmol();
  }, []);

  // Fetch PDB data when pdbId changes
  useEffect(() => {
    if (!pdbId) return;

    const fetchPdb = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const pdbUrl = getProteinPdbUrl(pdbId);
        const response = await fetch(pdbUrl);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch PDB: ${response.statusText}`);
        }

        const data = await response.text();
        setPdbData(data);
      } catch (err) {
        console.error('Failed to fetch PDB:', err);
        setError(
          err instanceof Error ? err.message : 'Failed to load PDB data'
        );
        setIsLoading(false);
      }
    };
    fetchPdb();
  }, [pdbId]);

  const getStyleForRepresentation = useCallback(
    (rep: ProteinRepresentation) => {
      switch (rep) {
        case 'cartoon':
          return { cartoon: { color: 'spectrum' } };
        case 'ball-and-stick':
          return { stick: { radius: 0.15 }, sphere: { scale: 0.25 } };
        case 'surface':
          return { surface: { opacity: 0.8, color: 'white' } };
        case 'ribbon':
          return { cartoon: { style: 'ribbon', color: 'spectrum' } };
        default:
          return { cartoon: { color: 'spectrum' } };
      }
    },
    []
  );

  const initViewer = useCallback(() => {
    if (!containerRef.current || !$3Dmol || !pdbData) return;

    setIsLoading(true);
    setError(null);

    try {
      // Clean up existing viewer
      if (viewerRef.current) {
        viewerRef.current.removeAllModels();
      }

      // Create viewer
      const viewer = $3Dmol.createViewer(containerRef.current, {
        backgroundColor: 'white',
      });
      viewerRef.current = viewer;

      // Add protein structure
      viewer.addModel(pdbData, 'pdb');
      viewer.setStyle({}, getStyleForRepresentation(representation));
      viewer.zoomTo();
      viewer.render();

      setIsLoading(false);
    } catch (err) {
      console.error('3D viewer error:', err);
      setError(
        err instanceof Error
          ? `Visualization error: ${err.message}`
          : 'Failed to render 3D structure'
      );
      setIsLoading(false);
    }
  }, [$3Dmol, pdbData, representation, getStyleForRepresentation]);

  useEffect(() => {
    initViewer();
  }, [$3Dmol, pdbData, initViewer]);

  // Update representation
  useEffect(() => {
    if (!viewerRef.current || !pdbData) return;
    try {
      viewerRef.current.setStyle({}, getStyleForRepresentation(representation));
      viewerRef.current.render();
    } catch (err) {
      console.error('Style update error:', err);
    }
  }, [representation, getStyleForRepresentation, pdbData]);

  const handleResetCamera = useCallback(() => {
    if (!viewerRef.current) return;
    viewerRef.current.zoomTo();
    viewerRef.current.render();
  }, []);

  if (error) {
    return (
      <Card className={`border-destructive bg-destructive/10 ${className}`}>
        <CardContent className="flex items-center gap-3 p-4">
          <AlertCircle className="size-5 text-destructive" />
          <div className="flex flex-col">
            <span className="text-sm font-medium text-destructive">
              Protein Visualization Error
            </span>
            <span className="text-xs text-muted-foreground">{error}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={`flex flex-col gap-3 ${className}`}>
      <div className="flex items-center gap-2">
        <Select
          value={representation}
          onValueChange={(value) =>
            setRepresentation(value as ProteinRepresentation)
          }
        >
          <SelectTrigger className="w-40">
            <SelectValue placeholder="Representation" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="cartoon">Cartoon</SelectItem>
            <SelectItem value="ball-and-stick">Ball & Stick</SelectItem>
            <SelectItem value="surface">Surface</SelectItem>
            <SelectItem value="ribbon">Ribbon</SelectItem>
          </SelectContent>
        </Select>
        <Button variant="outline" size="sm" onClick={handleResetCamera}>
          <RotateCcw className="mr-1 size-4" />
          Reset
        </Button>
      </div>

      <div className="relative">
        {isLoading && (
          <Skeleton
            className="absolute inset-0 z-10"
            style={{ width, height }}
          />
        )}
        <div
          ref={containerRef}
          style={{ width, height }}
          className={`rounded-lg border bg-background ${isLoading ? 'opacity-0' : 'opacity-100'} transition-opacity duration-200`}
        />
      </div>
    </div>
  );
}
