'use client';

import { AlertCircle, RotateCcw } from 'lucide-react';
import dynamic from 'next/dynamic';
import { useCallback, useState } from 'react';

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
import type { MoleculeRepresentation } from '@/types/visualization';

interface Molecule3DViewerProps {
  sdfData: string;
  width?: number;
  height?: number;
  className?: string;
  initialRepresentation?: MoleculeRepresentation;
}

// Lazy-load 3Dmol only on client (no SSR)
const Viewer3D = dynamic(() => import('./Viewer3D'), {
  ssr: false,
  loading: () => null, // we handle loading ourselves with Skeleton
});

export function Molecule3DViewer({
  sdfData,
  width = 400,
  height = 400,
  className,
  initialRepresentation = 'stick',
}: Molecule3DViewerProps) {
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [representation, setRepresentation] = useState<MoleculeRepresentation>(
    initialRepresentation,
  );

  // We'll show skeleton until both library + model are ready
  const handleReady = useCallback(() => {
    setIsLoading(false);
  }, []);

  const handleError = useCallback((msg: string) => {
    setError(msg);
    setIsLoading(false);
  }, []);

  if (error) {
    return (
      <Card className={`border-destructive bg-destructive/10 ${className}`}>
        <CardContent className="flex items-center gap-3 p-4">
          <AlertCircle className="text-destructive size-5" />
          <div className="flex flex-col">
            <span className="text-destructive text-sm font-medium">
              3D Visualization Error
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
          onValueChange={(v) => setRepresentation(v as MoleculeRepresentation)}
        >
          <SelectTrigger className="w-32">
            <SelectValue placeholder="Style" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="stick">Stick</SelectItem>
            <SelectItem value="sphere">Sphere</SelectItem>
            <SelectItem value="line">Line</SelectItem>
            <SelectItem value="cartoon">Cartoon</SelectItem>
          </SelectContent>
        </Select>

        <Button variant="outline" size="sm" onClick={() => { }}>
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

        <Viewer3D
          sdfData={sdfData}
          representation={representation}
          width={width}
          height={height}
          onReady={handleReady}
          onError={handleError}
        />
      </div>
    </div>
  );
}