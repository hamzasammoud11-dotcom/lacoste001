'use client';

import { ExternalLink, Search } from 'lucide-react';
import dynamic from 'next/dynamic';
import { useEffect, useMemo, useState } from 'react';

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
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Skeleton } from '@/components/ui/skeleton';
import { getProteinPdbUrl, getProteins } from '@/lib/api';
import type { Protein } from '@/types/visualization';

// Dynamic import of the protein viewer to prevent SSR issues
const ProteinViewer = dynamic(
  () => import('./_components/ProteinViewer').then((mod) => mod.ProteinViewer),
  {
    ssr: false,
    loading: () => <Skeleton className="size-[500px]" />,
  },
);

export default function Proteins3DPage() {
  const [proteins, setProteins] = useState<Protein[]>([]);
  const [selectedProtein, setSelectedProtein] = useState<Protein | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadProteins = async () => {
      try {
        setIsLoading(true);
        const data = await getProteins();
        setProteins(data);
        if (data.length > 0) {
          setSelectedProtein(data[0] || null);
        }
      } catch (err) {
        setError(
          err instanceof Error ? err.message : 'Failed to load proteins',
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
        p.description?.toLowerCase().includes(query),
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
    <div className="flex h-full gap-4 p-4">
      <Card className="w-80 shrink-0">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Proteins</CardTitle>
          <CardDescription>Select a protein to visualize</CardDescription>
        </CardHeader>
        <CardContent className="pb-3">
          <div className="relative">
            <Search className="text-muted-foreground absolute top-2.5 left-2.5 size-4" />
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
              <p className="text-muted-foreground p-4 text-center text-sm">
                No proteins found
              </p>
            ) : (
              <div className="space-y-1">
                {filteredProteins.map((protein) => (
                  <button
                    key={protein.id}
                    onClick={() => setSelectedProtein(protein)}
                    className={`hover:bg-accent w-full rounded-lg p-3 text-left transition-colors ${selectedProtein?.id === protein.id
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
                      <div className="text-muted-foreground mt-1 truncate text-xs">
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

      <div className="flex flex-1 flex-col gap-6">
        {selectedProtein ? (
          <>
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
                    <Badge variant="outline">
                      PDB: {selectedProtein.pdbId}
                    </Badge>
                    <Button variant="outline" size="sm" asChild>
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
                <div className="text-muted-foreground flex gap-4 text-sm">
                  <a
                    href={`https://www.rcsb.org/structure/${selectedProtein.pdbId}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-foreground flex items-center gap-1"
                  >
                    <ExternalLink className="size-3" />
                    View on RCSB PDB
                  </a>
                  <a
                    href={`https://www.uniprot.org/uniprotkb?query=${selectedProtein.name}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-foreground flex items-center gap-1"
                  >
                    <ExternalLink className="size-3" />
                    Search UniProt
                  </a>
                </div>
              </CardContent>
            </Card>

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
