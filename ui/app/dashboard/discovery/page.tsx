'use client';

import {
  AlertCircle,
  ArrowRight,
  Check,
  CheckCircle2,
  Circle,
  ImageIcon,
  Loader2,
  Maximize2,
  Microscope,
  Search,
  X,
} from 'lucide-react';
import * as React from 'react';

import {
  Checkbox,
  CheckboxIndicator,
} from '@/components/animate-ui/primitives/radix/checkbox';
import { PageHeader, SectionHeader } from '@/components/page-header';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { search } from '@/lib/api';
import { SearchResult } from '@/schemas/search';

export default function DiscoveryPage() {
  const [query, setQuery] = React.useState('');
  const [searchType, setSearchType] = React.useState('Similarity');
  const [database, setDatabase] = React.useState('both');
  const [includeImages, setIncludeImages] = React.useState(false);
  const [isSearching, setIsSearching] = React.useState(false);
  const [step, setStep] = React.useState(0);
  const [results, setResults] = React.useState<SearchResult[]>([]);
  const [error, setError] = React.useState<string | null>(null);

  const getApiType = (uiType: string, query: string): string => {
    const looksLikeSmiles = /^[A-Za-z0-9@+\-\[\]\(\)\\\/=#$.]+$/.test(
      query.trim(),
    );

    const looksLikeProtein =
      /^[ACDEFGHIKLMNPQRSTVWY]+$/i.test(query.trim()) && query.length > 20;

    if (uiType === 'Similarity' || uiType === 'Binding Affinity') {
      if (looksLikeSmiles && !looksLikeProtein) return 'drug';
      if (looksLikeProtein) return 'target';
    }
    return 'text';
  };

  const handleSearch = async () => {
    if (!query.trim()) return;

    setIsSearching(true);
    setStep(1);
    setError(null);
    setResults([]);

    try {
      setStep(1);

      await new Promise((r) => setTimeout(r, 300));
      setStep(2);

      const apiType = getApiType(searchType, query);
      const response = await search({
        query: query.trim(),
        type: apiType,
        limit: 10,
        dataset: database !== 'both' ? database.toLowerCase() : undefined,
        include_images: includeImages,
      });

      setStep(3);

      const data = response;

      await new Promise((r) => setTimeout(r, 200));
      setStep(4);
      setResults(data.results || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
      setStep(0);
    } finally {
      setIsSearching(false);
    }
  };

  const steps = [
    { name: 'Input', status: step > 0 ? 'done' : 'active' },
    {
      name: 'Encode',
      status: step > 1 ? 'done' : step === 1 ? 'active' : 'pending',
    },
    {
      name: 'Search',
      status: step > 2 ? 'done' : step === 2 ? 'active' : 'pending',
    },
    {
      name: 'Predict',
      status: step > 3 ? 'done' : step === 3 ? 'active' : 'pending',
    },
    { name: 'Results', status: step === 4 ? 'active' : 'pending' },
  ];

  return (
    <div className="animate-in fade-in space-y-8 duration-500">
      <PageHeader
        title="Drug Discovery"
        subtitle="Search for drug candidates using DeepPurpose + Qdrant"
        icon={<Microscope className="h-8 w-8" />}
      />

      <Card id="search">
        <div className="border-b p-4 font-semibold">Search Query</div>
        <CardContent className="p-6">
          <div className="grid grid-cols-1 gap-6 md:grid-cols-4">
            <div className="md:col-span-3">
              <Textarea
                placeholder={
                  searchType === 'Similarity'
                    ? 'Enter SMILES string (e.g., CC(=O)Nc1ccc(O)cc1 for Acetaminophen)'
                    : searchType === 'Binding Affinity'
                      ? 'Enter protein sequence (amino acids, e.g., MKKFFD...)'
                      : 'Enter drug name or keyword to search'
                }
                className="min-h-[120px] font-mono"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Search Type</Label>
                <Select value={searchType} onValueChange={setSearchType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Similarity">
                      Similarity (Drug SMILES)
                    </SelectItem>
                    <SelectItem value="Binding Affinity">
                      Binding Affinity (Protein)
                    </SelectItem>
                    <SelectItem value="Properties">
                      Properties (Text Search)
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Database</Label>
                <Select value={database} onValueChange={setDatabase}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select database" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="both">All Datasets</SelectItem>
                    <SelectItem value="kiba">
                      KIBA (Kinase Inhibitors)
                    </SelectItem>
                    <SelectItem value="davis">
                      DAVIS (Kinase Targets)
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center space-x-2 pt-2">
                <Checkbox
                  id="include-images"
                  checked={includeImages}
                  onCheckedChange={(c: boolean | "indeterminate") => setIncludeImages(!!c)}
                  className="peer flex h-4 w-4 shrink-0 items-center justify-center rounded-sm border border-primary ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground"
                >
                  <CheckboxIndicator className="flex items-center justify-center text-current">
                    <Check className="h-3 w-3" />
                  </CheckboxIndicator>
                </Checkbox>
                <label
                  htmlFor="include-images"
                  className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                >
                  Include Images
                </label>
              </div>

              <Button
                className="w-full"
                onClick={handleSearch}
                disabled={isSearching || !query}
              >
                {isSearching ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Search className="mr-2 h-4 w-4" />
                )}
                {isSearching ? 'Searching Qdrant...' : 'Search'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-destructive">
          <CardContent className="text-destructive flex items-center gap-3 p-4">
            <AlertCircle className="h-5 w-5" />
            <div>
              <div className="font-medium">Search Failed</div>
              <div className="text-sm">{error}</div>
              <div className="text-muted-foreground mt-1 text-xs">
                Make sure the API server is running: python -m uvicorn
                server.api:app --port 8000
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="space-y-4">
        <SectionHeader
          title="Pipeline Status"
          icon={<ArrowRight className="text-muted-foreground h-5 w-5" />}
        />

        <div className="relative">
          <div className="bg-muted absolute top-1/2 left-0 -z-10 h-0.5 w-full -translate-y-1/2 transform"></div>
          <div className="flex w-full items-center justify-between px-4">
            {steps.map((s, i) => (
              <div
                key={i}
                className="bg-background flex flex-col items-center gap-2 px-2"
              >
                <div
                  className={`flex h-8 w-8 items-center justify-center rounded-full border-2 transition-colors ${s.status === 'done'
                    ? 'bg-primary border-primary text-primary-foreground'
                    : s.status === 'active'
                      ? 'border-primary text-primary animate-pulse'
                      : 'border-muted text-muted-foreground bg-background'
                    }`}
                >
                  {s.status === 'done' ? (
                    <CheckCircle2 className="h-5 w-5" />
                  ) : (
                    <Circle className="h-5 w-5" />
                  )}
                </div>
                <span
                  className={`text-sm font-medium ${s.status === 'pending' ? 'text-muted-foreground' : 'text-foreground'}`}
                >
                  {s.name}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {step === 4 && results.length > 0 && (
        <div className="animate-in slide-in-from-bottom-4 space-y-4 duration-500">
          <SectionHeader
            title={`Results (${results.length} from Qdrant)`}
            icon={<CheckCircle2 className="h-5 w-5 text-green-500" />}
          />

          <Tabs defaultValue="candidates">
            <TabsList>
              <TabsTrigger value="candidates">Top Candidates</TabsTrigger>
              <TabsTrigger value="details">Raw Data</TabsTrigger>
            </TabsList>
            <TabsContent value="candidates" className="space-y-4">
              {results.map((result, i) => (
                <Card key={result.id}>
                  <CardContent className="flex items-center justify-between p-4">
                    <div className="flex-1">
                      <div className="mb-1 text-base font-semibold">
                        {result.metadata?.name || (result.modality === 'image' ? 'Image Result' : `Result ${i + 1}`)}
                      </div>
                      
                      {result.modality === 'image' ? (
                        <div className="flex items-start gap-4 py-2">
                          {/* If we have a URL or base64, try to show it. Otherwise show icon placeholder */}
                          {(() => {
                            // Check for base64 image (Priority 1)
                            const base64Img = result.metadata?.image;
                            if (typeof base64Img === 'string' && base64Img.startsWith('data:')) {
                              return (
                                <ExpandableImage 
                                    src={base64Img} 
                                    alt="Result" 
                                    caption={typeof result.metadata?.caption === 'string' ? result.metadata.caption : undefined}
                                />
                              );
                            }

                            // Check for direct URL or thumbnail URL (Priority 2 - e.g. IDR public URLs)
                            const directUrl = result.metadata?.thumbnail_url || result.metadata?.url;
                            if (typeof directUrl === 'string' && (directUrl.startsWith('http'))) {
                                // IDR thumbnail URLs are valid images. IDR 'url' is a webpage, so we prefer thumbnail_url if available.
                                // We'll try to use it if it looks like a thumbnail or if we have no other choice.
                                return (
                                <ExpandableImage 
                                    src={directUrl} 
                                    alt="Result (Remote)" 
                                    caption={typeof result.metadata?.caption === 'string' ? result.metadata.caption : undefined}
                                />
                                );
                            }

                            // Fallback: Data is missing (Stale record or ingestion failure)
                            return (
                              <div className="bg-muted flex h-24 w-24 flex-col items-center justify-center gap-1 rounded border p-1 text-center">
                                <ImageIcon className="text-muted-foreground h-6 w-6" />
                                <span className="text-[10px] text-muted-foreground leading-tight">Image Data<br/>Missing</span>
                              </div>
                            );
                          })()}
                          <div>
                            <div className="text-sm font-medium">{result.metadata?.description || "No description"}</div>
                            <div className="text-muted-foreground mt-1 text-xs break-all">
                              Source: {result.metadata?.source || "Upload"}
                              {typeof result.metadata?.caption === 'string' && result.metadata.caption.trim() !== '' ? (
                                <p className="mt-1 italic">{`"${result.metadata.caption}"`}</p>
                              ) : null}
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="text-muted-foreground font-mono text-sm">
                          {(result.metadata?.smiles || result.content)?.slice(
                            0,
                            60,
                          )}
                          {(result.metadata?.smiles || result.content)?.length >
                            60
                            ? '...'
                            : ''}
                        </div>
                      )}

                      {result.modality !== 'image' && result.metadata?.description && (
                        <div className="text-muted-foreground mt-1 text-sm">
                          {result.metadata.description}
                        </div>
                      )}
                      <div className="text-muted-foreground mt-2 flex gap-4 text-xs">
                        {result.metadata?.affinity_class && (
                          <span className="bg-muted rounded px-2 py-0.5">
                            Affinity: {result.metadata.affinity_class}
                          </span>
                        )}
                        {result.metadata?.label_true != null && (
                          <span className="bg-muted rounded px-2 py-0.5">
                            Label: {result.metadata.label_true.toFixed(2)}
                          </span>
                        )}
                        <span className="bg-muted rounded px-2 py-0.5">
                          {result.modality}
                        </span>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-muted-foreground text-sm">
                        Similarity
                      </div>
                      <div
                        className={`text-xl font-bold ${result.score >= 0.9
                          ? 'text-green-600'
                          : result.score >= 0.7
                            ? 'text-green-500'
                            : result.score >= 0.5
                              ? 'text-amber-500'
                              : 'text-muted-foreground'
                          }`}
                      >
                        {result.score.toFixed(3)}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </TabsContent>
            <TabsContent value="details">
              <Card>
                <CardContent className="p-4">
                  <pre className="bg-muted max-h-[400px] overflow-auto rounded p-4 text-xs">
                    {JSON.stringify(
                      results.map(r => ({
                        ...r,
                        metadata: {
                          ...r.metadata,
                          image: r.metadata?.image 
                            ? (String(r.metadata.image).length > 50 
                                ? `${String(r.metadata.image).substring(0, 50)}... [truncated]` 
                                : String(r.metadata.image))
                            : undefined
                        }
                      })), 
                      null, 
                      2
                    )}
                  </pre>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      )}

      {step === 4 && results.length === 0 && !error && (
        <Card>
          <CardContent className="text-muted-foreground p-8 text-center">
            No similar compounds found in Qdrant.
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function ExpandableImage({ src, alt, caption }: { src: string, alt: string, caption?: string }) {
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <>
      <div 
        className="group relative h-24 w-24 shrink-0 cursor-pointer overflow-hidden rounded-md border border-gray-200"
        onClick={() => setIsOpen(true)}
      >
        <img 
          src={src} 
          alt={alt} 
          className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-110"
          onError={(e) => { e.currentTarget.style.display = 'none'; }}
        />
        <div className="absolute inset-0 flex items-center justify-center bg-black/0 transition-colors group-hover:bg-black/20">
            <Maximize2 className="text-white opacity-0 transition-opacity group-hover:opacity-100 h-6 w-6" />
        </div>
      </div>

      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200" onClick={() => setIsOpen(false)}>
          <div className="relative max-h-[90vh] max-w-[90vw] overflow-auto rounded-lg bg-white p-2 shadow-2xl animate-in zoom-in-95 duration-200" onClick={e => e.stopPropagation()}>
             <button onClick={() => setIsOpen(false)} className="absolute right-2 top-2 z-10 rounded-full bg-white/80 p-1 hover:bg-white text-black transition-colors shadow-sm">
                <X className="h-5 w-5" />
             </button>
             <img src={src} alt={alt} className="max-h-[85vh] w-auto rounded" />
             {caption && <p className="mt-2 text-center text-sm text-gray-700 font-medium px-4 pb-2">{caption}</p>}
          </div>
        </div>
      )}
    </>
  );
}
