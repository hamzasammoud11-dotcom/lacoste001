'use client';

import { Check, Copy, Search } from 'lucide-react';
import dynamic from 'next/dynamic';
import { useCallback, useEffect, useMemo, useState } from 'react';

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';
import { getMolecules, getMoleculeSDF } from '@/lib/visualization-api';
import type { Molecule } from '@/lib/visualization-types';

// Dynamic import of the 3D viewer to prevent SSR issues
const Molecule3DViewer = dynamic(
  () =>
    import('./_components/Molecule3DViewer').then((mod) => mod.Molecule3DViewer),
  {
    ssr: false,
    loading: () => <Skeleton className="size-[400px]" />,
  }
);

export default function Molecules3DPage() {
  const [molecules, setMolecules] = useState<Molecule[]>([]);
  const [selectedMolecule, setSelectedMolecule] = useState<Molecule | null>(null);
  const [sdfData, setSdfData] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isSdfLoading, setIsSdfLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sdfError, setSdfError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Load molecules list
  useEffect(() => {
    const loadMolecules = async () => {
      try {
        setIsLoading(true);
        const data = await getMolecules();
        setMolecules(data);
        if (data.length > 0) {
          setSelectedMolecule(data[0]);
        }
      } catch (err) {
        setError(
          err instanceof Error ? err.message : 'Failed to load molecules'
        );
      } finally {
        setIsLoading(false);
      }
    };
    loadMolecules();
  }, []);

  // Load SDF when molecule changes
  useEffect(() => {
    if (!selectedMolecule) {
      setSdfData(null);
      return;
    }

    const loadSdf = async () => {
      try {
        setIsSdfLoading(true);
        setSdfError(null);
        const data = await getMoleculeSDF(selectedMolecule.id);
        setSdfData(data);
      } catch (err) {
        setSdfError(
          err instanceof Error ? err.message : 'Failed to load 3D structure'
        );
        setSdfData(null);
      } finally {
        setIsSdfLoading(false);
      }
    };
    loadSdf();
  }, [selectedMolecule]);

  const filteredMolecules = useMemo(() => {
    if (!searchQuery.trim()) return molecules;
    const query = searchQuery.toLowerCase();
    return molecules.filter(
      (m) =>
        m.name.toLowerCase().includes(query) ||
        m.smiles.toLowerCase().includes(query)
    );
  }, [molecules, searchQuery]);

  const handleCopySmiles = useCallback(async () => {
    if (!selectedMolecule) return;
    try {
      await navigator.clipboard.writeText(selectedMolecule.smiles);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      const textArea = document.createElement('textarea');
      textArea.value = selectedMolecule.smiles;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [selectedMolecule]);

  if (error) {
    return (
      <div className="flex h-full items-center justify-center">
        <Card className="max-w-md">
          <CardHeader>
            <CardTitle className="text-destructive">Error</CardTitle>
            <CardDescription>{error}</CardDescription>
          </CardHeader>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex h-full gap-4 p-4">
      {/* Left Panel - Molecule List */}
      <Card className="w-80 shrink-0">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Molecules</CardTitle>
          <CardDescription>Select a molecule for 3D view</CardDescription>
        </CardHeader>
        <CardContent className="pb-3">
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 size-4 text-muted-foreground" />
            <Input
              placeholder="Search molecules..."
              className="pl-8"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </CardContent>
        <Separator />
        <ScrollArea className="h-[calc(100vh-280px)]">
          <div className="p-2">
            {isLoading ? (
              <div className="space-y-2">
                {[1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-16 w-full" />
                ))}
              </div>
            ) : filteredMolecules.length === 0 ? (
              <p className="p-4 text-center text-sm text-muted-foreground">
                No molecules found
              </p>
            ) : (
              <div className="space-y-1">
                {filteredMolecules.map((molecule) => (
                  <button
                    key={molecule.id}
                    onClick={() => setSelectedMolecule(molecule)}
                    className={`w-full rounded-lg p-3 text-left transition-colors hover:bg-accent ${
                      selectedMolecule?.id === molecule.id
                        ? 'bg-accent'
                        : 'bg-transparent'
                    }`}
                  >
                    <div className="font-medium">{molecule.name}</div>
                    <div className="truncate text-xs text-muted-foreground">
                      {molecule.smiles}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </ScrollArea>
      </Card>

      {/* Right Panel - 3D Visualization */}
      <div className="flex flex-1 flex-col gap-6">
        {selectedMolecule ? (
          <>
            {/* Molecule Info Card */}
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle>{selectedMolecule.name}</CardTitle>
                    <CardDescription>
                      {selectedMolecule.description}
                    </CardDescription>
                  </div>
                  <Badge variant="outline">
                    PubChem: {selectedMolecule.pubchemCid}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-2">
                  <code className="flex-1 truncate rounded bg-muted px-2 py-1 text-sm">
                    {selectedMolecule.smiles}
                  </code>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleCopySmiles}
                    className="shrink-0"
                  >
                    {copied ? (
                      <>
                        <Check className="mr-1 size-4" />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="mr-1 size-4" />
                        Copy SMILES
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* 3D Structure Card */}
            <Card className="flex-1">
              <CardHeader>
                <CardTitle className="text-lg">3D Structure</CardTitle>
                <CardDescription>
                  Rotate: click + drag • Zoom: scroll • Pan: right-click + drag
                </CardDescription>
              </CardHeader>
              <CardContent className="flex items-center justify-center">
                {isSdfLoading ? (
                  <Skeleton className="size-[400px]" />
                ) : sdfError ? (
                  <Alert variant="destructive" className="max-w-md">
                    <AlertTitle>Failed to load 3D structure</AlertTitle>
                    <AlertDescription>{sdfError}</AlertDescription>
                  </Alert>
                ) : sdfData ? (
                  <Molecule3DViewer sdfData={sdfData} width={500} height={400} />
                ) : (
                  <p className="text-muted-foreground">
                    No 3D structure available
                  </p>
                )}
              </CardContent>
            </Card>
          </>
        ) : (
          <div className="flex flex-1 items-center justify-center">
            <p className="text-muted-foreground">
              Select a molecule to view its 3D structure
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
