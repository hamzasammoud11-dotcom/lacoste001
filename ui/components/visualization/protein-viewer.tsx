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
import { getProteinPDB } from '@/lib/api';
import type { ProteinRepresentation } from '@/types/visualization';

interface ProteinViewerProps {
  pdbId: string;
  width?: number;
  height?: number;
  className?: string;
  initialRepresentation?: ProteinRepresentation;
}


type $3Dmol = any;

export function ProteinViewer({
  pdbId,
  width = 500,
  height = 500,
  className,
  initialRepresentation = 'cartoon',
}: ProteinViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const viewerRef = useRef<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [representation, setRepresentation] = useState<ProteinRepresentation>(
    initialRepresentation,
  );
  const [$3Dmol, set$3Dmol] = useState<$3Dmol | null>(null);
  const [pdbData, setPdbData] = useState<string | null>(null);

  // Load 3Dmol.js dynamically
  useEffect(() => {
    const load3Dmol = async () => {
      try {

        const m = (await import('3dmol')) as any;
        set$3Dmol(m);
      } catch (err) {
        console.error('Failed to load 3Dmol:', err);
        setError('Failed to load 3D visualization library');
        setIsLoading(false);
      }
    };
    load3Dmol();
  }, []);

  useEffect(() => {
    if (!pdbId) return;

    const fetchPdb = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const data = await getProteinPDB(pdbId);
        setPdbData(data);
      } catch (err) {
        console.error('Failed to fetch PDB:', err);
        setError(
          err instanceof Error ? err.message : 'Failed to load PDB data',
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
          return { surface: {} }; // Surface handled via addSurface
        case 'ribbon':
          return { cartoon: { style: 'trace', color: 'spectrum' } };
        default:
          return { cartoon: { color: 'spectrum' } };
      }
    },
    [],
  );

  const initViewer = useCallback(() => {
    if (!containerRef.current || !$3Dmol || !pdbData) return;

    setIsLoading(true);
    setError(null);

    try {
      if (viewerRef.current) {
        viewerRef.current.removeAllModels();
      }

      const viewer = $3Dmol.createViewer(containerRef.current, {
        backgroundColor: '#0a0a0a',
      });
      viewerRef.current = viewer;

      viewer.addModel(pdbData, 'pdb');

      if (representation === 'surface') {
        viewer.setStyle({}, { cartoon: { color: 'spectrum' } });
        viewer.addSurface(
          $3Dmol.SurfaceType.VDW,
          {
            opacity: 0.85,
            color: 'spectrum',
          },
          {},
          {},
        );
      } else {
        viewer.setStyle({}, getStyleForRepresentation(representation));
      }

      viewer.zoomTo();
      viewer.render();

      setIsLoading(false);
    } catch (err) {
      console.error('3D viewer error:', err);
      setError(
        err instanceof Error
          ? `Visualization error: ${err.message}`
          : 'Failed to render 3D structure',
      );
      setIsLoading(false);
    }
  }, [$3Dmol, pdbData, representation, getStyleForRepresentation]);

  useEffect(() => {
    initViewer();
  }, [$3Dmol, pdbData, initViewer]);

  useEffect(() => {
    if (!viewerRef.current || !pdbData || !$3Dmol) return;
    try {
      viewerRef.current.removeAllSurfaces();

      if (representation === 'surface') {
        viewerRef.current.setStyle(
          {},
          { cartoon: { color: 'spectrum', opacity: 0.5 } },
        );
        viewerRef.current.addSurface(
          $3Dmol.SurfaceType.VDW,
          {
            opacity: 0.85,
            color: 'spectrum',
          },
          {},
          {},
        );
      } else {
        viewerRef.current.setStyle(
          {},
          getStyleForRepresentation(representation),
        );
      }

      viewerRef.current.render();
    } catch (err) {
      console.error('Style update error:', err);
    }
  }, [representation, getStyleForRepresentation, pdbData, $3Dmol]);

  const handleResetCamera = useCallback(() => {
    if (!viewerRef.current) return;
    viewerRef.current.zoomTo();
    viewerRef.current.render();
  }, []);

  if (error) {
    return (
      <Card className={`border-destructive bg-destructive/10 ${className}`}>
        <CardContent className="flex items-center gap-3 p-4">
          <AlertCircle className="text-destructive size-5" />
          <div className="flex flex-col">
            <span className="text-destructive text-sm font-medium">
              Protein Visualization Error
            </span>
            <span className="text-muted-foreground text-xs">{error}</span>
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
          className={`bg-background rounded-lg border ${isLoading ? 'opacity-0' : 'opacity-100'} transition-opacity duration-200`}
        />
      </div>
    </div>
  );
}
