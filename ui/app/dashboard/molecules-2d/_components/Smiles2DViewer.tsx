'use client';

import { AlertCircle } from 'lucide-react';
import { useCallback, useEffect, useId, useRef, useState } from 'react';
import SmilesDrawer from 'smiles-drawer';

import { Card, CardContent } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';

interface Smiles2DViewerProps {
  smiles: string;
  width?: number;
  height?: number;
  className?: string;
}

export function Smiles2DViewer({
  smiles,
  width = 400,
  height = 300,
  className,
}: Smiles2DViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasId = useId();
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const drawMolecule = useCallback(async () => {
    if (!canvasRef.current || !smiles) {
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const drawer = new SmilesDrawer.SmiDrawer({
        width,
        height,
        bondThickness: 1.5,
        bondLength: 15,
        shortBondLength: 0.85,
        bondSpacing: 4,
        atomVisualization: 'default',
        isomeric: true,
        debug: false,
        terminalCarbons: false,
        explicitHydrogens: false,
        compactDrawing: true,
        fontSizeLarge: 11,
        fontSizeSmall: 8,
        padding: 20,
      });

      try {
        await new Promise<void>((resolve, reject) => {
          (drawer as any).draw(
            smiles,
            `[id="${canvasId}"]`,
            'light',
            () => resolve(),
            (drawError: Error) => reject(drawError)
          );
        });
      } catch (drawError) {
        throw drawError;
      }

      setIsLoading(false);
    } catch (err) {
      console.error('SMILES drawing error:', err);
      setError(
        err instanceof Error
          ? `Invalid SMILES: ${err.message}`
          : 'Failed to render molecule structure'
      );
      setIsLoading(false);
    }
  }, [smiles, width, height, canvasId]);

  useEffect(() => {
    drawMolecule();
  }, [drawMolecule]);

  if (error) {
    return (
      <Card className={`border-destructive bg-destructive/10 ${className}`}>
        <CardContent className="flex items-center gap-3 p-4">
          <AlertCircle className="size-5 text-destructive" />
          <div className="flex flex-col">
            <span className="text-sm font-medium text-destructive">
              Visualization Error
            </span>
            <span className="text-xs text-muted-foreground">{error}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={`relative ${className}`}>
      {isLoading && (
        <Skeleton
          className="absolute inset-0"
          style={{ width, height }}
        />
      )}
      <canvas
        ref={canvasRef}
        id={canvasId}
        width={width}
        height={height}
        className={`bg-background rounded-lg ${isLoading ? 'opacity-0' : 'opacity-100'} transition-opacity duration-200`}
      />
    </div>
  );
}
