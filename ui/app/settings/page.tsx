"use client"

import { PageHeader, SectionHeader } from "@/components/page-header"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Settings, Brain, Save } from "lucide-react"

export default function SettingsPage() {
  return (
    <div className="space-y-8 animate-in fade-in duration-500">
         <PageHeader
            title="Settings"
            subtitle="Configure models, databases, and preferences"
            icon={<Settings className="h-8 w-8 text-primary" />} 
        />

        <Tabs defaultValue="models" className="w-full">
            <TabsList className="w-full justify-start overflow-x-auto">
                <TabsTrigger value="models">Models</TabsTrigger>
                <TabsTrigger value="database">Database</TabsTrigger>
                <TabsTrigger value="api">API Keys</TabsTrigger>
                <TabsTrigger value="appearance">Appearance</TabsTrigger>
                <TabsTrigger value="system">System</TabsTrigger>
            </TabsList>
            <TabsContent value="models" className="space-y-6 mt-6">
                 <SectionHeader title="Model Configuration" icon={<Brain className="h-5 w-5 text-primary" />} />
                 
                 <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card>
                        <CardHeader>
                            <CardTitle>Embedding Models</CardTitle>
                            <CardDescription>Configure models used for molecular and protein embeddings</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="mol-encoder">Molecule Encoder</Label>
                                <Select defaultValue="MolCLR">
                                    <SelectTrigger id="mol-encoder">
                                        <SelectValue placeholder="Select molecule encoder" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="MolCLR">MolCLR (Recommended)</SelectItem>
                                        <SelectItem value="ChemBERTa">ChemBERTa</SelectItem>
                                        <SelectItem value="GraphMVP">GraphMVP</SelectItem>
                                        <SelectItem value="MolBERT">MolBERT</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="prot-encoder">Protein Encoder</Label>
                                <Select defaultValue="ESM-2">
                                    <SelectTrigger id="prot-encoder">
                                        <SelectValue placeholder="Select protein encoder" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="ESM-2">ESM-2 (Recommended)</SelectItem>
                                        <SelectItem value="ProtTrans">ProtTrans</SelectItem>
                                        <SelectItem value="UniRep">UniRep</SelectItem>
                                        <SelectItem value="SeqVec">SeqVec</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle>Prediction Heads</CardTitle>
                            <CardDescription>Configure downstream task predictors</CardDescription>
                        </CardHeader>
                         <CardContent className="space-y-4">
                             <div className="space-y-2">
                                <Label htmlFor="binding">Binding Predictor</Label>
                                <Select defaultValue="DrugBAN">
                                    <SelectTrigger id="binding">
                                        <SelectValue placeholder="Select predictor" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="DrugBAN">DrugBAN (Recommended)</SelectItem>
                                        <SelectItem value="DeepDTA">DeepDTA</SelectItem>
                                        <SelectItem value="GraphDTA">GraphDTA</SelectItem>
                                        <SelectItem value="Custom">Custom</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                             <div className="space-y-2">
                                <Label htmlFor="property">Property Predictor</Label>
                                <Select defaultValue="ADMET-AI">
                                    <SelectTrigger id="property">
                                        <SelectValue placeholder="Select predictor" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="ADMET-AI">ADMET-AI (Recommended)</SelectItem>
                                        <SelectItem value="ChemProp">ChemProp</SelectItem>
                                        <SelectItem value="Custom">Custom</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                        </CardContent>
                    </Card>
                 </div>

                 <Card>
                    <CardHeader>
                        <CardTitle>LLM Settings</CardTitle>
                        <CardDescription>Configure language models for evidence retrieval and reasoning</CardDescription>
                    </CardHeader>
                    <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-2">
                            <Label htmlFor="llm-provider">LLM Provider</Label>
                            <Select defaultValue="OpenAI">
                                <SelectTrigger id="llm-provider">
                                    <SelectValue placeholder="Select provider" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="OpenAI">OpenAI</SelectItem>
                                    <SelectItem value="Anthropic">Anthropic</SelectItem>
                                    <SelectItem value="Local">Local (Ollama)</SelectItem>
                                    <SelectItem value="Azure">Azure OpenAI</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                         <div className="space-y-2">
                             <Label htmlFor="llm-model">Model</Label>
                             <Select defaultValue="GPT-4o">
                                <SelectTrigger id="llm-model">
                                    <SelectValue placeholder="Select model" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="GPT-4o">GPT-4o</SelectItem>
                                    <SelectItem value="GPT-4-turbo">GPT-4-turbo</SelectItem>
                                    <SelectItem value="Claude 3.5">Claude 3.5 Sonnet</SelectItem>
                                    <SelectItem value="Llama 3">Llama 3.1 70B</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                        <div className="col-span-1 md:col-span-2 space-y-4">
                            <div className="space-y-2">
                                <div className="flex items-center justify-between">
                                    <Label>Temperature: 0.7</Label>
                                    <span className="text-xs text-muted-foreground">Creativity vs Precision</span>
                                </div>
                                <Slider defaultValue={[0.7]} max={1} step={0.1} />
                            </div>
                            <div className="flex items-center space-x-2">
                                <Switch id="stream" defaultChecked />
                                <Label htmlFor="stream">Stream Responses</Label>
                            </div>
                        </div>
                    </CardContent>
                 </Card>
            </TabsContent>
            
            <TabsContent value="appearance">
                <Card>
                     <CardContent className="p-12 text-center text-muted-foreground">
                        Theme settings coming soon.
                    </CardContent>
                </Card>
            </TabsContent>
            
             <TabsContent value="database">
                <Card>
                     <CardContent className="p-12 text-center text-muted-foreground">
                        Database connection settings.
                    </CardContent>
                </Card>
            </TabsContent>
            
            <TabsContent value="api">
                <Card>
                     <CardContent className="p-12 text-center text-muted-foreground">
                        API Key configuration.
                    </CardContent>
                </Card>
            </TabsContent>
        </Tabs>

        <div className="fixed bottom-6 right-6">
            <Button size="lg" className="shadow-2xl">
                <Save className="mr-2 h-4 w-4" />
                Save Changes
            </Button>
        </div>
    </div>
  )
}
