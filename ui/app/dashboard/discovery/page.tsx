"use client"

import { ArrowRight,CheckCircle2, Circle, Loader2, Microscope, Search } from "lucide-react"
import * as React from "react"

import { PageHeader, SectionHeader } from "@/components/page-header"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent,TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"

interface Candidate {
  name?: string;
  smiles?: string;
  content?: string;
  score: number;
  mw?: number | string;
  logp?: number | string;
}

interface DiscoveryResult {
  candidates: Candidate[];
  query: string;
  search_type: string;
}

export default function DiscoveryPage() {
  const [query, setQuery] = React.useState("")
  const [searchType, setSearchType] = React.useState("similarity")
  const [database, setDatabase] = React.useState("all")
  const [isSearching, setIsSearching] = React.useState(false)
  const [step, setStep] = React.useState(0)
  const [jobId, setJobId] = React.useState<string | null>(null)
  const [results, setResults] = React.useState<DiscoveryResult | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  const handleSearch = async () => {
    setIsSearching(true)
    setStep(1)
    setError(null)
    setResults(null)
    
    try {
      // Start discovery pipeline
      const response = await fetch('/api/discovery', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          searchType,
          database,
          limit: 10,
        }),
      })
      
      const data = await response.json()
      
      if (data.job_id) {
        setJobId(data.job_id)
        setStep(2)
        
        // Poll for results
        pollForResults(data.job_id)
      } else if (data.candidates) {
        // Direct result (sync mode)
        setResults({ candidates: data.candidates, query, search_type: searchType })
        setStep(4)
        setIsSearching(false)
      }
    } catch (err) {
      console.error("Discovery error:", err)
      setError(`Discovery failed: ${err instanceof Error ? err.message : 'Unknown error'}. Ensure the backend is running.`)
      setIsSearching(false)
      setStep(0)
    }
  }
  
  const pollForResults = async (id: string) => {
    const maxAttempts = 30
    let attempts = 0
    
    const poll = async () => {
      try {
        const response = await fetch(`/api/discovery?jobId=${id}`)
        const data = await response.json()
        
        // Update step based on current_step
        if (data.current_step === "encode") setStep(2)
        else if (data.current_step === "search") setStep(3)
        else if (data.current_step === "predict") setStep(3)
        
        if (data.status === "completed") {
          setResults(data.result)
          setStep(4)
          setIsSearching(false)
          return
        } else if (data.status === "failed") {
          setError(data.error || "Pipeline failed")
          setIsSearching(false)
          return
        }
        
        attempts++
        if (attempts < maxAttempts) {
          setTimeout(poll, 1000)
        } else {
          setError("Pipeline timeout")
          setIsSearching(false)
        }
      } catch (err) {
        setError("Failed to get status")
        setIsSearching(false)
      }
    }
    
    poll()
  }

  const steps = [
    { name: "Input", status: step > 0 ? "done" : "active" },
    { name: "Encode", status: step > 1 ? "done" : (step === 1 ? "active" : "pending") },
    { name: "Search", status: step > 2 ? "done" : (step === 2 ? "active" : "pending") },
    { name: "Predict", status: step > 3 ? "done" : (step === 3 ? "active" : "pending") },
    { name: "Results", status: step === 4 ? "active" : "pending" },
  ]

  const candidates = results?.candidates || []

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <PageHeader 
        title="Drug Discovery" 
        subtitle="Search for drug candidates with AI-powered analysis" 
        icon={<Microscope className="h-8 w-8" />}
      />

      <Card>
        <div className="p-4 border-b font-semibold">Search Query</div>
        <CardContent className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="md:col-span-3">
                    <Textarea 
                        placeholder="Enter a natural language query, SMILES string, or FASTA sequence..."
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
                                <SelectItem value="similarity">Similarity</SelectItem>
                                <SelectItem value="binding">Binding Affinity</SelectItem>
                                <SelectItem value="properties">Properties</SelectItem>
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
                                <SelectItem value="all">All</SelectItem>
                                <SelectItem value="drugbank">DrugBank</SelectItem>
                                <SelectItem value="chembl">ChEMBL</SelectItem>
                                <SelectItem value="zinc">ZINC</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                    <Button 
                        className="w-full" 
                        onClick={handleSearch}
                        disabled={isSearching || !query}
                    >
                        {isSearching ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
                        {isSearching ? "Running..." : "Search"}
                    </Button>
                </div>
            </div>
        </CardContent>
      </Card>

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

      {error && (
        <Card className="border-amber-500 bg-amber-50 dark:bg-amber-950">
          <CardContent className="p-4 text-amber-700 dark:text-amber-300">
            ⚠️ {error}
          </CardContent>
        </Card>
      )}

      {step === 4 && candidates.length > 0 && (
        <div className="space-y-4 animate-in slide-in-from-bottom-4 duration-500">
             <SectionHeader title="Results" icon={<CheckCircle2 className="h-5 w-5 text-green-500" />} />
             
             <Tabs defaultValue="candidates">
                <TabsList>
                    <TabsTrigger value="candidates">Top Candidates ({candidates.length})</TabsTrigger>
                    <TabsTrigger value="analysis">Property Analysis</TabsTrigger>
                    <TabsTrigger value="evidence">Evidence</TabsTrigger>
                </TabsList>
                <TabsContent value="candidates" className="space-y-4">
                    {candidates.map((candidate, i) => (
                        <Card key={i}>
                            <CardContent className="p-4 flex items-center justify-between">
                                <div>
                                    <div className="font-bold text-lg">
                                      {candidate.name || `Candidate ${i + 1}`}
                                    </div>
                                    <div className="text-sm text-muted-foreground font-mono mb-1">
                                      {candidate.smiles || candidate.content || "N/A"}
                                    </div>
                                    <div className="flex gap-4 text-sm text-muted-foreground">
                                        <span>MW: {candidate.mw ?? "N/A"}</span>
                                        <span>LogP: {candidate.logp ?? "N/A"}</span>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <div className="text-sm text-muted-foreground">Score</div>
                                    <div className={`text-xl font-bold ${
                                        candidate.score >= 0.9 ? 'text-green-600' : 
                                        candidate.score >= 0.8 ? 'text-green-500' : 'text-amber-500'
                                    }`}>
                                        {typeof candidate.score === 'number' ? candidate.score.toFixed(2) : candidate.score}
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    ))}
                </TabsContent>
                <TabsContent value="analysis">
                    <Card>
                        <CardContent className="p-12 text-center text-muted-foreground">
                            Chart visualization would go here (using Recharts).
                        </CardContent>
                    </Card>
                </TabsContent>
                <TabsContent value="evidence">
                     <Card>
                        <CardContent className="p-12 text-center text-muted-foreground">
                            Evidence graph visualization would go here.
                        </CardContent>
                    </Card>
                </TabsContent>
             </Tabs>
        </div>
      )}
    </div>
  )
}
