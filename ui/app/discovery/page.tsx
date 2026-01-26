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

export default function DiscoveryPage() {
  const [query, setQuery] = React.useState("")
  const [isSearching, setIsSearching] = React.useState(false)
  const [step, setStep] = React.useState(0)

  const handleSearch = () => {
    setIsSearching(true)
    setStep(1)
    
    // Simulate pipeline
    setTimeout(() => setStep(2), 1500)
    setTimeout(() => setStep(3), 3000)
    setTimeout(() => {
        setStep(4)
        setIsSearching(false)
    }, 4500)
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
                        <Select defaultValue="Similarity">
                            <SelectTrigger>
                                <SelectValue placeholder="Select type" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="Similarity">Similarity</SelectItem>
                                <SelectItem value="Binding Affinity">Binding Affinity</SelectItem>
                                <SelectItem value="Properties">Properties</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                     <div className="space-y-2">
                        <Label>Database</Label>
                        <Select defaultValue="All">
                            <SelectTrigger>
                                <SelectValue placeholder="Select database" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="All">All</SelectItem>
                                <SelectItem value="DrugBank">DrugBank</SelectItem>
                                <SelectItem value="ChEMBL">ChEMBL</SelectItem>
                                <SelectItem value="ZINC">ZINC</SelectItem>
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

      {step === 4 && (
        <div className="space-y-4 animate-in slide-in-from-bottom-4 duration-500">
             <SectionHeader title="Results" icon={<CheckCircle2 className="h-5 w-5 text-green-500" />} />
             
             <Tabs defaultValue="candidates">
                <TabsList>
                    <TabsTrigger value="candidates">Top Candidates</TabsTrigger>
                    <TabsTrigger value="analysis">Property Analysis</TabsTrigger>
                    <TabsTrigger value="evidence">Evidence</TabsTrigger>
                </TabsList>
                <TabsContent value="candidates" className="space-y-4">
                    {[
                        { name: "Candidate A", score: 0.95, mw: "342.4", logp: "2.1" },
                        { name: "Candidate B", score: 0.89, mw: "298.3", logp: "1.8" },
                        { name: "Candidate C", score: 0.82, mw: "415.5", logp: "3.2" },
                        { name: "Candidate D", score: 0.76, mw: "267.3", logp: "1.5" },
                        { name: "Candidate E", score: 0.71, mw: "389.4", logp: "2.8" },
                    ].map((candidate, i) => (
                        <Card key={i}>
                            <CardContent className="p-4 flex items-center justify-between">
                                <div>
                                    <div className="font-bold text-lg">{candidate.name}</div>
                                    <div className="flex gap-4 text-sm text-muted-foreground">
                                        <span>MW: {candidate.mw}</span>
                                        <span>LogP: {candidate.logp}</span>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <div className="text-sm text-muted-foreground">Score</div>
                                    <div className={`text-xl font-bold ${
                                        candidate.score >= 0.9 ? 'text-green-600' : 
                                        candidate.score >= 0.8 ? 'text-green-500' : 'text-amber-500'
                                    }`}>
                                        {candidate.score}
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
