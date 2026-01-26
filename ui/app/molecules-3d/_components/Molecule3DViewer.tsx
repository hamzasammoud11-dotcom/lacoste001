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
import type { MoleculeRepresentation } from '@/lib/visualization-types';

interface Molecule3DViewerProps {
  sdfData: string;
  width?: number;
  height?: number;
  className?: string;
  initialRepresentation?: MoleculeRepresentation;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type $3Dmol = any;

export function Molecule3DViewer({
  sdfData,
  width = 400,
  height = 400,
  className,
  initialRepresentation = 'stick',
}: Molecule3DViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const viewerRef = useRef<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [representation, setRepresentation] =
    useState<MoleculeRepresentation>(initialRepresentation);
  const [$3Dmol, set$3Dmol] = useState<$3Dmol | null>(null);

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

  const getStyleForRepresentation = useCallback(
    (rep: MoleculeRepresentation) => {
      switch (rep) {
        case 'stick':
          return { stick: { radius: 0.15 } };
        case 'sphere':
          return { sphere: { scale: 0.3 } };
        case 'line':
          return { line: { linewidth: 2 } };
        case 'cartoon':
          return { cartoon: {} };
        default:
          return { stick: { radius: 0.15 } };
      }
    },
    []
  );

  const initViewer = useCallback(() => {
    if (!containerRef.current || !$3Dmol || !sdfData) return;

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

      // Add molecule
      viewer.addModel(sdfData, 'sdf');
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
  }, [$3Dmol, sdfData, representation, getStyleForRepresentation]);

  useEffect(() => {
    initViewer();
  }, [$3Dmol, sdfData, initViewer]);

  // Update representation
  useEffect(() => {
    if (!viewerRef.current) return;
    try {
      viewerRef.current.setStyle({}, getStyleForRepresentation(representation));
      viewerRef.current.render();
    } catch (err) {
      console.error('Style update error:', err);
    }
  }, [representation, getStyleForRepresentation]);

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
              3D Visualization Error
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
            setRepresentation(value as MoleculeRepresentation)
          }
        >
          <SelectTrigger className="w-32">
            <SelectValue placeholder="Style" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="stick">Stick</SelectItem>
            <SelectItem value="sphere">Sphere</SelectItem>
            <SelectItem value="line">Line</SelectItem>
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
