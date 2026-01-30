'use client';

import { AlertCircle, FlaskConical } from 'lucide-react';
import { useCallback, useEffect, useId, useRef, useState } from 'react';
import SmilesDrawer from 'smiles-drawer';

import { Skeleton } from '@/components/ui/skeleton';

interface Smiles2DViewerProps {
  smiles: string;
  width?: number;
  height?: number;
  className?: string;
  useFallback?: boolean; // Use NIH Cactus API as fallback
}

/**
 * Validates that input is a non-empty string that looks like SMILES.
 * Returns null if invalid, or the cleaned string if valid.
 * CRITICAL: This must handle ANY input type safely to prevent .split() errors in smiles-drawer
 */
function validateSmilesInput(input: unknown): string | null {
  // Must be a string - reject objects, arrays, numbers, etc.
  if (input === null || input === undefined) {
    return null;
  }
  
  if (typeof input !== 'string') {
    // If it's an object with a toString, try to use it (but be careful)
    if (typeof input === 'object') {
      // Objects like { smiles: 'CCO' } should not be accepted
      return null;
    }
    // Try to convert numbers to string as a last resort
    if (typeof input === 'number') {
      return null; // Numbers are not valid SMILES
    }
    return null;
  }
  
  const trimmed = input.trim();
  
  // Must not be empty
  if (!trimmed || trimmed.length === 0) {
    return null;
  }
  
  // Reject strings that look like JSON objects or arrays
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    return null;
  }
  
  // Basic sanity check - SMILES should contain at least one letter
  if (!/[A-Za-z]/.test(trimmed)) {
    return null;
  }
  
  // Additional check: SMILES shouldn't have spaces (except for salts with .)
  // But some valid SMILES have spaces in extended formats, so just warn
  
  return trimmed;
}

export function Smiles2DViewer({
  smiles,
  width = 400,
  height = 300,
  className,
  useFallback = true,
}: Smiles2DViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasId = useId();
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [useNihFallback, setUseNihFallback] = useState(false);

  // CRITICAL: Validate smiles input before any processing
  const smilesString = validateSmilesInput(smiles);

  const drawMolecule = useCallback(async () => {
    // CRITICAL: Early return if smiles is invalid
    if (!smilesString) {
      setIsLoading(false);
      setError('Invalid or empty SMILES string');
      return;
    }
    
    if (!canvasRef.current) {
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

      await new Promise<void>((resolve, reject) => {
        // Extra safety: ensure smilesString is definitely a string before passing to parser
        const safeSmiles = String(smilesString);
        if (!safeSmiles || safeSmiles.length === 0) {
          reject(new Error('Empty SMILES string'));
          return;
        }
        
        try {
          SmilesDrawer.parse(
            safeSmiles,
            (tree: any) => {
              try {
                drawer.draw(tree, canvasRef.current, 'light');
                resolve();
              } catch (drawErr) {
                reject(drawErr);
              }
            },
            (parseErr: any) => reject(parseErr),
          );
        } catch (parseError) {
          reject(parseError);
        }
      });

      setIsLoading(false);
      setUseNihFallback(false);
    } catch (err) {
      console.error('SMILES drawing error:', err);
      // Try NIH Cactus fallback if enabled
      if (useFallback) {
        setUseNihFallback(true);
        setIsLoading(false);
        setError(null);
      } else {
        setError(
          err instanceof Error
            ? `Invalid SMILES: ${err.message}`
            : 'Failed to render molecule structure',
        );
        setIsLoading(false);
      }
    }
  }, [smilesString, width, height, useFallback]);

  useEffect(() => {
    drawMolecule();
  }, [drawMolecule]);

  // CRITICAL: If smiles is invalid, show placeholder immediately
  if (!smilesString) {
    return (
      <div 
        className={`flex items-center justify-center bg-muted/50 rounded-lg border border-dashed ${className}`}
        style={{ width, height }}
      >
        <div className="flex flex-col items-center gap-1 text-muted-foreground">
          <FlaskConical className="h-6 w-6 opacity-50" />
          <span className="text-xs">Invalid Structure</span>
        </div>
      </div>
    );
  }

  // Use NIH Cactus API fallback if SmilesDrawer failed
  if (useNihFallback && smilesString) {
    const encodedSmiles = encodeURIComponent(smilesString);
    const nihUrl = `https://cactus.nci.nih.gov/chemical/structure/${encodedSmiles}/image?width=${width}&height=${height}`;
    
    return (
      <div className={`relative ${className}`} style={{ width, height }}>
        <img
          src={nihUrl}
          alt="Molecule structure"
          width={width}
          height={height}
          className="bg-white rounded-lg"
          onError={() => {
            setUseNihFallback(false);
            setError('Failed to render molecule (both SmilesDrawer and NIH API failed)');
          }}
          onLoad={() => setIsLoading(false)}
        />
        {isLoading && (
          <Skeleton className="absolute inset-0" style={{ width, height }} />
        )}
      </div>
    );
  }

  if (error) {
    return (
      <div 
        className={`flex items-center justify-center bg-destructive/10 rounded-lg border border-destructive/30 ${className}`}
        style={{ width, height }}
      >
        <div className="flex flex-col items-center gap-1 p-2 text-center">
          <AlertCircle className="text-destructive h-5 w-5" />
          <span className="text-destructive text-xs font-medium">Error</span>
          <span className="text-muted-foreground text-[10px] max-w-[90%] truncate">{error}</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      {isLoading && (
        <Skeleton className="absolute inset-0" style={{ width, height }} />
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
