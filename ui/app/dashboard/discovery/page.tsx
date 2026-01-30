'use client';

import {
  AlertCircle,
  ArrowRight,
  Beaker,
  Check,
  CheckCircle2,
  Circle,
  Compass,
  FileText,
  ImageIcon,
  Loader2,
  Maximize2,
  Microscope,
  Search,
  Sparkles,
  Upload,
  X,
} from 'lucide-react';
import * as React from 'react';

import {
  Checkbox,
  CheckboxIndicator,
} from '@/components/animate-ui/primitives/radix/checkbox';
import { DesignAssistantModal } from '@/components/dashboard/design/design-assistant-modal';
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
import { Slider } from '@/components/ui/slider';
import { search, searchByImage, searchNeighbors, searchExperiments } from '@/lib/api';
import { SearchResult } from '@/schemas/search';

type SearchMode = 'text' | 'image' | 'experiment';

export default function DiscoveryPage() {
  const [query, setQuery] = React.useState('');
  const [searchMode, setSearchMode] = React.useState<SearchMode>('text');
  const [searchType, setSearchType] = React.useState('Similarity');
  const [database, setDatabase] = React.useState('both');
  const [includeImages, setIncludeImages] = React.useState(false);
  const [isSearching, setIsSearching] = React.useState(false);
  const [step, setStep] = React.useState(0);
  const [results, setResults] = React.useState<SearchResult[]>([]);
  const [error, setError] = React.useState<string | null>(null);
  
  // Image upload state
  const [uploadedImage, setUploadedImage] = React.useState<string | null>(null);
  const [imageFileName, setImageFileName] = React.useState<string>('');
  const [isDragging, setIsDragging] = React.useState(false);
  
  // Experiment filters - use sentinel values to avoid empty string in Select
  const [experimentType, setExperimentType] = React.useState<string>('__any__');
  const [outcome, setOutcome] = React.useState<string>('__any__');
  const [qualityMin, setQualityMin] = React.useState<number>(0);
  
  // Neighbor exploration state
  const [neighborResults, setNeighborResults] = React.useState<any>(null);
  const [isLoadingNeighbors, setIsLoadingNeighbors] = React.useState(false);
  const [selectedItemId, setSelectedItemId] = React.useState<string | null>(null);
  const [selectedItemName, setSelectedItemName] = React.useState<string | null>(null);

  // Design assistant state
  const [designModalOpen, setDesignModalOpen] = React.useState(false);
  const [designSourceItem, setDesignSourceItem] = React.useState<string>('');
  const [designSourceModality, setDesignSourceModality] = React.useState<'molecule' | 'protein' | 'auto'>('auto');
  const [designSourceName, setDesignSourceName] = React.useState<string>('');

  const handleSuggestVariants = (result: SearchResult) => {
    // Get the content to use as source for variants
    const content = result.metadata?.smiles || result.content || '';
    const modality = result.modality === 'drug' || result.modality === 'molecule' 
      ? 'molecule' 
      : result.modality === 'target' || result.modality === 'protein'
        ? 'protein'
        : 'auto';
    const name = result.metadata?.name as string || '';
    
    setDesignSourceItem(content);
    setDesignSourceModality(modality);
    setDesignSourceName(name);
    setDesignModalOpen(true);
  };

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

  // File upload handlers
  const handleFileUpload = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (PNG, JPG, etc.)');
      return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
      const base64 = e.target?.result as string;
      setUploadedImage(base64);
      setImageFileName(file.name);
      setError(null);
    };
    reader.onerror = () => {
      setError('Failed to read file');
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const clearUploadedImage = () => {
    setUploadedImage(null);
    setImageFileName('');
  };

  // Explore neighbors handler
  const handleExploreNeighbors = async (itemId: string, itemName?: string) => {
    setIsLoadingNeighbors(true);
    setSelectedItemId(itemId);
    setSelectedItemName(itemName || null);
    try {
      const response = await searchNeighbors({
        item_id: itemId,
        top_k: 10,
        include_cross_modal: true,
        diversity: 0.3,
      });
      setNeighborResults(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load neighbors');
    } finally {
      setIsLoadingNeighbors(false);
    }
  };

  const handleSearch = async () => {
    // Validate input based on mode
    if (searchMode === 'text' && !query.trim()) return;
    if (searchMode === 'image' && !uploadedImage) return;
    if (searchMode === 'experiment' && !query.trim()) return;

    setIsSearching(true);
    setStep(1);
    setError(null);
    setResults([]);
    setNeighborResults(null);

    try {
      setStep(1);
      await new Promise((r) => setTimeout(r, 300));
      setStep(2);

      let data: { results: SearchResult[] };

      if (searchMode === 'image' && uploadedImage) {
        // Image search
        data = await searchByImage({
          image: uploadedImage,
          image_type: 'other',
          top_k: 10,
          use_mmr: true,
        });
      } else if (searchMode === 'experiment') {
        // Experiment search - convert sentinel values to undefined
        const expType = experimentType === '__any__' ? undefined : experimentType;
        const expOutcome = outcome === '__any__' ? undefined : outcome;
        const expResponse = await searchExperiments({
          query: query.trim(),
          experiment_type: expType,
          outcome: expOutcome,
          quality_min: qualityMin > 0 ? qualityMin : undefined,
          top_k: 10,
        });
        // Convert experiment results to SearchResult format
        data = {
          results: expResponse.experiments.map((exp) => ({
            id: exp.id,
            score: exp.score,
            content: exp.description || exp.title,
            modality: 'experiment',
            metadata: {
              name: exp.title,
              experiment_id: exp.experiment_id,
              experiment_type: exp.experiment_type,
              outcome: exp.outcome,
              quality_score: exp.quality_score,
              measurements: exp.measurements,
              conditions: exp.conditions,
              target: exp.target,
              molecule: exp.molecule,
            },
          })),
        };
      } else {
        // Text search (original)
        const apiType = getApiType(searchType, query);
        data = await search({
          query: query.trim(),
          type: apiType,
          limit: 10,
          dataset: database !== 'both' ? database.toLowerCase() : undefined,
          include_images: includeImages,
        });
      }

      setStep(3);
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
          {/* Search Mode Selector */}
          <div className="mb-6">
            <Label className="mb-2 block">Search Mode</Label>
            <div className="flex gap-2">
              <Button
                variant={searchMode === 'text' ? 'default' : 'outline'}
                onClick={() => setSearchMode('text')}
                className="flex-1"
              >
                <Search className="mr-2 h-4 w-4" />
                Text / SMILES
              </Button>
              <Button
                variant={searchMode === 'image' ? 'default' : 'outline'}
                onClick={() => setSearchMode('image')}
                className="flex-1"
              >
                <ImageIcon className="mr-2 h-4 w-4" />
                Image
              </Button>
              <Button
                variant={searchMode === 'experiment' ? 'default' : 'outline'}
                onClick={() => setSearchMode('experiment')}
                className="flex-1"
              >
                <Beaker className="mr-2 h-4 w-4" />
                Experiment
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-6 md:grid-cols-4">
            <div className="md:col-span-3">
              {/* Text Search Input */}
              {searchMode === 'text' && (
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
              )}

              {/* Image Upload Dropzone */}
              {searchMode === 'image' && (
                <div
                  className={`min-h-[120px] rounded-lg border-2 border-dashed p-6 transition-colors ${
                    isDragging
                      ? 'border-primary bg-primary/5'
                      : 'border-muted-foreground/25 hover:border-primary/50'
                  }`}
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                >
                  {uploadedImage ? (
                    <div className="flex items-center gap-4">
                      <img
                        src={uploadedImage}
                        alt="Uploaded"
                        className="h-24 w-24 rounded-lg object-cover"
                      />
                      <div className="flex-1">
                        <p className="font-medium">{imageFileName}</p>
                        <p className="text-muted-foreground text-sm">
                          Ready to search
                        </p>
                      </div>
                      <Button variant="ghost" size="icon" onClick={clearUploadedImage}>
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center text-center">
                      <Upload className="text-muted-foreground mb-2 h-10 w-10" />
                      <p className="text-muted-foreground mb-2">
                        Drag & drop an image here, or click to select
                      </p>
                      <input
                        type="file"
                        accept="image/*"
                        className="hidden"
                        id="image-upload"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) handleFileUpload(file);
                        }}
                      />
                      <Button variant="outline" asChild>
                        <label htmlFor="image-upload" className="cursor-pointer">
                          Select Image
                        </label>
                      </Button>
                      <p className="text-muted-foreground mt-2 text-xs">
                        Supports: microscopy, gels, spectra, structures
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Experiment Search Input */}
              {searchMode === 'experiment' && (
                <div className="space-y-4">
                  <Textarea
                    placeholder="Search experiments (e.g., 'EGFR binding assay', 'kinase inhibition')"
                    className="min-h-[80px]"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <Label className="mb-1 text-xs">Experiment Type</Label>
                      <Select value={experimentType} onValueChange={(v) => setExperimentType(v === '__any__' ? '' : v)}>
                        <SelectTrigger className="h-8 text-xs">
                          <SelectValue placeholder="Any type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="__any__">Any type</SelectItem>
                          <SelectItem value="binding_assay">Binding Assay</SelectItem>
                          <SelectItem value="activity_assay">Activity Assay</SelectItem>
                          <SelectItem value="admet">ADMET</SelectItem>
                          <SelectItem value="phenotypic">Phenotypic</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label className="mb-1 text-xs">Outcome</Label>
                      <Select value={outcome} onValueChange={(v) => setOutcome(v === '__any__' ? '' : v)}>
                        <SelectTrigger className="h-8 text-xs">
                          <SelectValue placeholder="Any outcome" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="__any__">Any outcome</SelectItem>
                          <SelectItem value="success">Success</SelectItem>
                          <SelectItem value="failure">Failure</SelectItem>
                          <SelectItem value="partial">Partial</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label className="mb-1 text-xs">Min Quality: {qualityMin.toFixed(1)}</Label>
                      <Slider
                        value={[qualityMin]}
                        onValueChange={([v]) => setQualityMin(v)}
                        min={0}
                        max={1}
                        step={0.1}
                        className="mt-2"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div className="space-y-4">
              {searchMode === 'text' && (
                <>
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
                </>
              )}

              <Button
                className="w-full"
                onClick={handleSearch}
                disabled={
                  isSearching || 
                  (searchMode === 'text' && !query) ||
                  (searchMode === 'image' && !uploadedImage) ||
                  (searchMode === 'experiment' && !query)
                }
              >
                {isSearching ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : searchMode === 'image' ? (
                  <ImageIcon className="mr-2 h-4 w-4" />
                ) : searchMode === 'experiment' ? (
                  <Beaker className="mr-2 h-4 w-4" />
                ) : (
                  <Search className="mr-2 h-4 w-4" />
                )}
                {isSearching 
                  ? 'Searching...' 
                  : searchMode === 'image' 
                    ? 'Search by Image' 
                    : searchMode === 'experiment'
                      ? 'Search Experiments'
                      : 'Search'}
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
              {/* Show Photos tab if include_images was checked and we have image results */}
              {includeImages && results.some(r => r.modality === 'image') && (
                <TabsTrigger value="photos">
                  Photos ({results.filter(r => r.modality === 'image').length})
                </TabsTrigger>
              )}
              <TabsTrigger value="details">Raw Data</TabsTrigger>
            </TabsList>
            <TabsContent value="candidates" className="space-y-4">
              {results.filter(r => r.modality !== 'image' || !includeImages).map((result, i) => (
                <Card key={result.id}>
                  <CardContent className="flex items-center justify-between p-4">
                    <div className="flex-1">
                      <div className="mb-1 text-base font-semibold">
                        {result.metadata?.name || (result.modality === 'image' ? 'Image Result' : result.modality === 'experiment' ? (result.metadata?.title || `Experiment ${i + 1}`) : `Result ${i + 1}`)}
                      </div>
                      
                      {/* Experiment Result Display */}
                      {result.modality === 'experiment' ? (
                        <div className="space-y-2">
                          <div className="flex flex-wrap gap-2 text-xs">
                            {result.metadata?.experiment_type && (
                              <span className="bg-blue-100 text-blue-800 rounded px-2 py-0.5">
                                {result.metadata.experiment_type}
                              </span>
                            )}
                            {result.metadata?.outcome && (
                              <span className={`rounded px-2 py-0.5 ${
                                result.metadata.outcome === 'success' 
                                  ? 'bg-green-100 text-green-800'
                                  : result.metadata.outcome === 'failure'
                                    ? 'bg-red-100 text-red-800'
                                    : 'bg-amber-100 text-amber-800'
                              }`}>
                                {result.metadata.outcome}
                              </span>
                            )}
                            {result.metadata?.quality_score != null && (
                              <span className="bg-muted rounded px-2 py-0.5">
                                Quality: {(result.metadata.quality_score as number).toFixed(2)}
                              </span>
                            )}
                          </div>
                          {result.metadata?.target && (
                            <div className="text-muted-foreground text-sm">
                              Target: <span className="font-medium">{result.metadata.target}</span>
                            </div>
                          )}
                          {result.metadata?.molecule && (
                            <div className="text-muted-foreground font-mono text-xs">
                              {(result.metadata.molecule as string).slice(0, 50)}
                              {(result.metadata.molecule as string).length > 50 ? '...' : ''}
                            </div>
                          )}
                          {/* Measurements */}
                          {Array.isArray(result.metadata?.measurements) && result.metadata.measurements.length > 0 && (
                            <div className="bg-muted/50 mt-2 rounded p-2">
                              <div className="text-xs font-medium mb-1">Measurements:</div>
                              <div className="flex flex-wrap gap-2">
                                {(result.metadata.measurements as Array<{name: string; value: number; unit: string}>).map((m, idx) => (
                                  <span key={idx} className="bg-background rounded px-2 py-0.5 text-xs">
                                    {m.name}: {m.value} {m.unit}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : result.modality === 'image' ? (
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

                      {result.modality !== 'image' && result.modality !== 'experiment' && result.metadata?.description && (
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
                    <div className="flex flex-col items-end gap-2">
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
                      {/* Action Buttons */}
                      <div className="flex gap-2">
                        {/* Suggest Variants Button - Only for molecules and proteins */}
                        {(result.modality === 'drug' || result.modality === 'molecule' || 
                          result.modality === 'target' || result.modality === 'protein') && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleSuggestVariants(result)}
                            className="text-xs border-purple-500/30 hover:bg-purple-500/10 hover:text-purple-600"
                          >
                            <Sparkles className="mr-1 h-3 w-3" />
                            Suggest Variants
                          </Button>
                        )}
                        {/* Explore Neighbors Button */}
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleExploreNeighbors(result.id, result.metadata?.name as string || result.metadata?.smiles as string || result.content?.slice(0, 30))}
                          disabled={isLoadingNeighbors && selectedItemId === result.id}
                          className="text-xs"
                        >
                          {isLoadingNeighbors && selectedItemId === result.id ? (
                            <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                          ) : (
                            <Compass className="mr-1 h-3 w-3" />
                          )}
                          Explore Neighbors
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </TabsContent>
            
            {/* Photos Tab Content */}
            {includeImages && (
              <TabsContent value="photos" className="space-y-4">
                {results.filter(r => r.modality === 'image').length === 0 ? (
                  <Card>
                    <CardContent className="p-8 text-center text-muted-foreground">
                      <ImageIcon className="mx-auto h-12 w-12 mb-4 opacity-50" />
                      <p>No photo results found.</p>
                      <p className="text-sm mt-2">Try a different search query or check if images have been ingested.</p>
                    </CardContent>
                  </Card>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {results.filter(r => r.modality === 'image').map((result, i) => (
                      <Card key={result.id} className="overflow-hidden">
                        <CardContent className="p-0">
                          <div className="aspect-video w-full bg-muted relative">
                            {(() => {
                              const base64Img = result.metadata?.image;
                              const directUrl = result.metadata?.thumbnail_url || result.metadata?.url;
                              
                              if (typeof base64Img === 'string' && base64Img.startsWith('data:')) {
                                return (
                                  <ExpandableImage 
                                    src={base64Img} 
                                    alt={`Image ${i + 1}`}
                                    caption={typeof result.metadata?.caption === 'string' ? result.metadata.caption : undefined}
                                  />
                                );
                              }
                              
                              if (typeof directUrl === 'string' && directUrl.startsWith('http')) {
                                return (
                                  <ExpandableImage 
                                    src={directUrl} 
                                    alt={`Image ${i + 1}`}
                                    caption={typeof result.metadata?.caption === 'string' ? result.metadata.caption : undefined}
                                  />
                                );
                              }
                              
                              return (
                                <div className="flex h-full w-full items-center justify-center">
                                  <ImageIcon className="h-12 w-12 text-muted-foreground/50" />
                                </div>
                              );
                            })()}
                          </div>
                          <div className="p-4">
                            <div className="flex items-start justify-between mb-2">
                              <span className="text-sm font-medium truncate flex-1">
                                {result.metadata?.description || result.metadata?.caption || `Image ${i + 1}`}
                              </span>
                              <span className={`ml-2 text-sm font-bold ${
                                result.score >= 0.9 ? 'text-green-600' :
                                result.score >= 0.7 ? 'text-green-500' :
                                'text-amber-500'
                              }`}>
                                {result.score.toFixed(3)}
                              </span>
                            </div>
                            <div className="text-xs text-muted-foreground space-y-1">
                              {result.metadata?.source && (
                                <p>Source: {result.metadata.source}</p>
                              )}
                              {result.metadata?.image_type && (
                                <p>Type: {result.metadata.image_type}</p>
                              )}
                            </div>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleExploreNeighbors(result.id, result.metadata?.description as string || result.metadata?.caption as string || `Image ${i + 1}`)}
                              disabled={isLoadingNeighbors && selectedItemId === result.id}
                              className="w-full mt-3 text-xs"
                            >
                              {isLoadingNeighbors && selectedItemId === result.id ? (
                                <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                              ) : (
                                <Compass className="mr-1 h-3 w-3" />
                              )}
                              Explore Similar
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </TabsContent>
            )}
            
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

      {/* Neighbor Exploration Results */}
      {neighborResults && (
        <div className="animate-in slide-in-from-bottom-4 space-y-4 duration-300">
          <SectionHeader
            title={`Neighbors of ${selectedItemName || selectedItemId?.slice(0, 8) + '...'} (${neighborResults.total_found} found)`}
            icon={<Compass className="h-5 w-5 text-blue-500" />}
          />
          
          {/* Facets */}
          {neighborResults.facets && Object.keys(neighborResults.facets).length > 0 && (
            <div className="flex flex-wrap gap-2 mb-4">
              {Object.entries(neighborResults.facets).map(([modality, count]) => (
                <span key={modality} className="bg-muted rounded-full px-3 py-1 text-xs font-medium">
                  {modality}: {count as number}
                </span>
              ))}
            </div>
          )}
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {neighborResults.neighbors.slice(0, 9).map((neighbor: any, i: number) => (
              <Card key={neighbor.id} className="hover:shadow-md transition-shadow">
                <CardContent className="p-4">
                  <div className="flex items-start justify-between mb-2">
                    <span className="bg-muted rounded px-2 py-0.5 text-xs">
                      {neighbor.modality}
                    </span>
                    <span className={`text-sm font-bold ${
                      neighbor.score >= 0.9 ? 'text-green-600' :
                      neighbor.score >= 0.7 ? 'text-green-500' :
                      'text-amber-500'
                    }`}>
                      {neighbor.score.toFixed(3)}
                    </span>
                  </div>
                  <div className="text-sm font-medium truncate">
                    {neighbor.metadata?.name || neighbor.metadata?.title || neighbor.metadata?.description || `Neighbor ${i + 1}`}
                  </div>
                  {/* Render image if modality is image */}
                  {neighbor.modality === 'image' ? (
                    <div className="mt-2">
                      {(() => {
                        const base64Img = neighbor.metadata?.image;
                        const directUrl = neighbor.metadata?.thumbnail_url || neighbor.metadata?.url;
                        if (typeof base64Img === 'string' && base64Img.startsWith('data:')) {
                          return (
                            <img 
                              src={base64Img} 
                              alt={neighbor.metadata?.description || 'Neighbor image'}
                              className="w-full h-24 object-cover rounded"
                            />
                          );
                        }
                        if (typeof directUrl === 'string' && directUrl.startsWith('http')) {
                          return (
                            <img 
                              src={directUrl} 
                              alt={neighbor.metadata?.description || 'Neighbor image'}
                              className="w-full h-24 object-cover rounded"
                            />
                          );
                        }
                        return (
                          <div className="w-full h-24 bg-muted rounded flex items-center justify-center">
                            <ImageIcon className="h-8 w-8 text-muted-foreground" />
                          </div>
                        );
                      })()}
                    </div>
                  ) : (
                    <div className="text-muted-foreground text-xs truncate mt-1">
                      {neighbor.content?.slice(0, 80)}
                      {neighbor.content?.length > 80 ? '...' : ''}
                    </div>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="mt-2 w-full text-xs"
                    onClick={() => handleExploreNeighbors(neighbor.id, neighbor.metadata?.name || neighbor.metadata?.title || neighbor.metadata?.description)}
                  >
                    <Compass className="mr-1 h-3 w-3" />
                    Explore from here
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
          
          <Button variant="outline" onClick={() => setNeighborResults(null)} className="w-full">
            <X className="mr-2 h-4 w-4" />
            Close Neighbors
          </Button>
        </div>
      )}

      {step === 4 && results.length === 0 && !error && (
        <Card>
          <CardContent className="text-muted-foreground p-8 text-center">
            No similar compounds found in Qdrant.
          </CardContent>
        </Card>
      )}

      {/* Design Assistant Modal */}
      <DesignAssistantModal
        isOpen={designModalOpen}
        onClose={() => setDesignModalOpen(false)}
        sourceItem={designSourceItem}
        sourceModality={designSourceModality}
        sourceName={designSourceName}
      />
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
