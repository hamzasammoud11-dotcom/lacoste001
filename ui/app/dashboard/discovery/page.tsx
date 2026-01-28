"use client"

import { AlertCircle,ArrowRight,CheckCircle2, Circle, Loader2, Microscope, Search } from "lucide-react"
import * as React from "react"

import { PageHeader, SectionHeader } from "@/components/page-header"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent,TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface SearchResult {
  id: string;
  score: number;
  mmr_score?: number;
  content: string;
  modality: string;
  metadata: {
    name?: string;
    smiles?: string;
    description?: string;
    source?: string;
    label_true?: number;
    affinity_class?: string;
    [key: string]: unknown;
  };
}

export default function DiscoveryPage() {
  const [query, setQuery] = React.useState("")
  const [searchType, setSearchType] = React.useState("Similarity")
  const [database, setDatabase] = React.useState("both")
  const [isSearching, setIsSearching] = React.useState(false)
  const [step, setStep] = React.useState(0)
  const [results, setResults] = React.useState<SearchResult[]>([])
  const [error, setError] = React.useState<string | null>(null)

  // Map UI search type to API type
  const getApiType = (uiType: string, query: string): string => {
    // If it looks like SMILES (contains chemistry chars), use drug encoding
    const looksLikeSmiles = /^[A-Za-z0-9@+\-\[\]\(\)\\\/=#$.]+$/.test(query.trim())
    // If it looks like protein sequence (all caps amino acids)
    const looksLikeProtein = /^[ACDEFGHIKLMNPQRSTVWY]+$/i.test(query.trim()) && query.length > 20
    
    if (uiType === "Similarity" || uiType === "Binding Affinity") {
      if (looksLikeSmiles && !looksLikeProtein) return "drug"
      if (looksLikeProtein) return "target"
      return "text" // Fallback to text search
    }
    return "text"
  }

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setIsSearching(true)
    setStep(1)
    setError(null)
    setResults([])
    
    try {
      // Step 1: Input received
      setStep(1)
      
      // Step 2: Determine type and encode
      await new Promise(r => setTimeout(r, 300))
      setStep(2)
      
      const apiType = getApiType(searchType, query)
      
      // Step 3: Actually search Qdrant via our API
      const response = await fetch(`${API_BASE}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          type: apiType,
          limit: 10,
          dataset: database !== "both" ? database.toLowerCase() : undefined
        })
      });
      
      setStep(3)
      
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `API error: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Step 4: Process results
      await new Promise(r => setTimeout(r, 200))
      setStep(4)
      setResults(data.results || [])
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
      setStep(0)
    } finally {
      setIsSearching(false)
    }
  }

  const steps = [
    { name: "Input", status: step > 0 ? "done" : "active" },
    { name: "Encode", status: step > 1 ? "done" : (step === 1 ? "active" : "pending") },
    { name: "Search", status: step > 2 ? "done" : (step === 2 ? "active" : "pending") },
    { name: "Predict", status: step > 3 ? "done" : (step === 3 ? "active" : "pending") },
    { name: "Results", status: step === 4 ? "active" : "pending" },
  ]

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <PageHeader 
        title="Drug Discovery" 
        subtitle="Search for drug candidates using DeepPurpose + Qdrant" 
        icon={<Microscope className="h-8 w-8" />}
      />

      <Card id="search">
        <div className="p-4 border-b font-semibold">Search Query</div>
        <CardContent className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="md:col-span-3">
                    <Textarea 
                        placeholder={
                          searchType === "Similarity" 
                            ? "Enter SMILES string (e.g., CC(=O)Nc1ccc(O)cc1 for Acetaminophen)" 
                            : searchType === "Binding Affinity"
                            ? "Enter protein sequence (amino acids, e.g., MKKFFD...)"
                            : "Enter drug name or keyword to search"
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
                                <SelectItem value="Similarity">Similarity (Drug SMILES)</SelectItem>
                                <SelectItem value="Binding Affinity">Binding Affinity (Protein)</SelectItem>
                                <SelectItem value="Properties">Properties (Text Search)</SelectItem>
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
                                <SelectItem value="kiba">KIBA (Kinase Inhibitors)</SelectItem>
                                <SelectItem value="davis">DAVIS (Kinase Targets)</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                    <Button 
                        className="w-full" 
                        onClick={handleSearch}
                        disabled={isSearching || !query}
                    >
                        {isSearching ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
                        {isSearching ? "Searching Qdrant..." : "Search"}
                    </Button>
                </div>
            </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-destructive">
          <CardContent className="p-4 flex items-center gap-3 text-destructive">
            <AlertCircle className="h-5 w-5" />
            <div>
              <div className="font-medium">Search Failed</div>
              <div className="text-sm">{error}</div>
              <div className="text-xs mt-1 text-muted-foreground">
                Make sure the API server is running: python -m uvicorn server.api:app --port 8001
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="space-y-4">
         <SectionHeader title="Pipeline Status" icon={<ArrowRight className="h-5 w-5 text-muted-foreground" />} />
         
         <div className="relative">
            <div className="absolute left-0 top-1/2 w-full h-0.5 bg-muted -z-10 transform -translate-y-1/2"></div>
            <div className="flex justify-between items-center w-full px-4">
                {steps.map((s, i) => (
                    <div key={i} className="flex flex-col items-center gap-2 bg-background px-2">
                        <div className={`h-8 w-8 rounded-full flex items-center justify-center border-2 transition-colors ${
                            s.status === 'done' ? 'bg-primary border-primary text-primary-foreground' : 
                            s.status === 'active' ? 'border-primary text-primary animate-pulse' : 'border-muted text-muted-foreground bg-background'
                        }`}>
                            {s.status === 'done' ? <CheckCircle2 className="h-5 w-5" /> : <Circle className="h-5 w-5" />}
                        </div>
                        <span className={`text-sm font-medium ${s.status === 'pending' ? 'text-muted-foreground' : 'text-foreground'}`}>
                            {s.name}
                        </span>
                    </div>
                ))}
            </div>
         </div>
      </div>

      {step === 4 && results.length > 0 && (
        <div className="space-y-4 animate-in slide-in-from-bottom-4 duration-500">
             <SectionHeader title={`Results (${results.length} from Qdrant)`} icon={<CheckCircle2 className="h-5 w-5 text-green-500" />} />
             
             <Tabs defaultValue="candidates">
                <TabsList>
                    <TabsTrigger value="candidates">Top Candidates</TabsTrigger>
                    <TabsTrigger value="details">Raw Data</TabsTrigger>
                </TabsList>
                <TabsContent value="candidates" className="space-y-4">
                    {results.map((result, i) => (
                        <Card key={result.id}>
                            <CardContent className="p-4 flex items-center justify-between">
                                <div className="flex-1">
                                    <div className="font-semibold text-base mb-1">
                                      {result.metadata?.name || `Result ${i + 1}`}
                                    </div>
                                    <div className="font-mono text-sm text-muted-foreground">
                                      {(result.metadata?.smiles || result.content)?.slice(0, 60)}
                                      {(result.metadata?.smiles || result.content)?.length > 60 ? '...' : ''}
                                    </div>
                                    {result.metadata?.description && (
                                      <div className="text-sm text-muted-foreground mt-1">
                                        {result.metadata.description}
                                      </div>
                                    )}
                                    <div className="flex gap-4 text-xs text-muted-foreground mt-2">
                                        {result.metadata?.affinity_class && (
                                          <span className="bg-muted px-2 py-0.5 rounded">
                                            Affinity: {result.metadata.affinity_class}
                                          </span>
                                        )}
                                        {result.metadata?.label_true != null && (
                                          <span className="bg-muted px-2 py-0.5 rounded">
                                            Label: {result.metadata.label_true.toFixed(2)}
                                          </span>
                                        )}
                                        <span className="bg-muted px-2 py-0.5 rounded">
                                          {result.modality}
                                        </span>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <div className="text-sm text-muted-foreground">Similarity</div>
                                    <div className={`text-xl font-bold ${
                                        result.score >= 0.9 ? 'text-green-600' : 
                                        result.score >= 0.7 ? 'text-green-500' : 
                                        result.score >= 0.5 ? 'text-amber-500' : 'text-muted-foreground'
                                    }`}>
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
                            <pre className="text-xs overflow-auto max-h-[400px] bg-muted p-4 rounded">
                              {JSON.stringify(results, null, 2)}
                            </pre>
                        </CardContent>
                    </Card>
                </TabsContent>
             </Tabs>
        </div>
      )}
      
      {step === 4 && results.length === 0 && !error && (
        <Card>
          <CardContent className="p-8 text-center text-muted-foreground">
            No similar compounds found in Qdrant.
          </CardContent>
        </Card>
      )}
    </div>
  )
}
