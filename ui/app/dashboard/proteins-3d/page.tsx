'use client';

import { ExternalLink, Search } from 'lucide-react';
import dynamic from 'next/dynamic';
import { useEffect, useMemo, useState } from 'react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';
import { getProteins, getProteinPdbUrl } from '@/lib/visualization-api';
import type { Protein } from '@/lib/visualization-types';

// Dynamic import of the protein viewer to prevent SSR issues
const ProteinViewer = dynamic(
  () =>
    import('./_components/ProteinViewer').then((mod) => mod.ProteinViewer),
  {
    ssr: false,
    loading: () => <Skeleton className="size-[500px]" />,
  }
);

export default function Proteins3DPage() {
  const [proteins, setProteins] = useState<Protein[]>([]);
  const [selectedProtein, setSelectedProtein] = useState<Protein | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load proteins list
  useEffect(() => {
    const loadProteins = async () => {
      try {
        setIsLoading(true);
        const data = await getProteins();
        setProteins(data);
        if (data.length > 0) {
          setSelectedProtein(data[0]);
        }
      } catch (err) {
        setError(
          err instanceof Error ? err.message : 'Failed to load proteins'
        );
      } finally {
        setIsLoading(false);
      }
    };
    loadProteins();
  }, []);

  const filteredProteins = useMemo(() => {
    if (!searchQuery.trim()) return proteins;
    const query = searchQuery.toLowerCase();
    return proteins.filter(
      (p) =>
        p.name.toLowerCase().includes(query) ||
        p.pdbId.toLowerCase().includes(query) ||
        p.description?.toLowerCase().includes(query)
    );
  }, [proteins, searchQuery]);

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
    <div className="flex h-full gap-6 p-6">
      {/* Left Panel - Protein List */}
      <Card className="w-80 shrink-0">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Proteins</CardTitle>
          <CardDescription>Select a protein to visualize</CardDescription>
        </CardHeader>
        <CardContent className="pb-3">
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 size-4 text-muted-foreground" />
            <Input
              placeholder="Search proteins..."
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
            ) : filteredProteins.length === 0 ? (
              <p className="p-4 text-center text-sm text-muted-foreground">
                No proteins found
              </p>
            ) : (
              <div className="space-y-1">
                {filteredProteins.map((protein) => (
                  <button
                    key={protein.id}
                    onClick={() => setSelectedProtein(protein)}
                    className={`w-full rounded-lg p-3 text-left transition-colors hover:bg-accent ${
                      selectedProtein?.id === protein.id
                        ? 'bg-accent'
                        : 'bg-transparent'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{protein.name}</span>
                      <Badge variant="secondary" className="text-xs">
                        {protein.pdbId}
                      </Badge>
                    </div>
                    {protein.description && (
                      <div className="mt-1 truncate text-xs text-muted-foreground">
                        {protein.description}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </ScrollArea>
      </Card>

      {/* Right Panel - 3D Visualization */}
      <div className="flex flex-1 flex-col gap-6">
        {selectedProtein ? (
          <>
            {/* Protein Info Card */}
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle>{selectedProtein.name}</CardTitle>
                    <CardDescription>
                      {selectedProtein.description}
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">PDB: {selectedProtein.pdbId}</Badge>
                    <Button
                      variant="outline"
                      size="sm"
                      asChild
                    >
                      <a
                        href={getProteinPdbUrl(selectedProtein.id)}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <ExternalLink className="mr-1 size-4" />
                        Open PDB
                      </a>
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex gap-4 text-sm text-muted-foreground">
                  <a
                    href={`https://www.rcsb.org/structure/${selectedProtein.pdbId}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1 hover:text-foreground"
                  >
                    <ExternalLink className="size-3" />
                    View on RCSB PDB
                  </a>
                  <a
                    href={`https://www.uniprot.org/uniprotkb?query=${selectedProtein.name}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1 hover:text-foreground"
                  >
                    <ExternalLink className="size-3" />
                    Search UniProt
                  </a>
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
                <ProteinViewer
                  key={selectedProtein.pdbId}
                  pdbId={selectedProtein.pdbId}
                  width={600}
                  height={500}
                />
              </CardContent>
            </Card>
          </>
        ) : (
          <div className="flex flex-1 items-center justify-center">
            <p className="text-muted-foreground">
              Select a protein to view its 3D structure
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
