'use client';

import { Brain, Save, Settings } from 'lucide-react';
import { useEffect, useState } from 'react';

import { PageHeader, SectionHeader } from '@/components/page-header';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export default function SettingsPage() {
  const [molEncoder, setMolEncoder] = useState('MolCLR');
  const [protEncoder, setProtEncoder] = useState('ESM-2');
  const [bindingPredictor, setBindingPredictor] = useState('DrugBAN');
  const [propertyPredictor, setPropertyPredictor] = useState('ADMET-AI');
  const [llmProvider, setLlmProvider] = useState('Local');
  const [llmModel, setLlmModel] = useState('Llama3');
  const [temperature, setTemperature] = useState(0.7);
  const [stream, setStream] = useState(true);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem('bioflow_settings');
    if (!saved) return;
    try {
      const data = JSON.parse(saved);
      setMolEncoder(data.molEncoder || 'MolCLR');
      setProtEncoder(data.protEncoder || 'ESM-2');
      setBindingPredictor(data.bindingPredictor || 'DrugBAN');
      setPropertyPredictor(data.propertyPredictor || 'ADMET-AI');
      setLlmProvider(data.llmProvider || 'Local');
      setLlmModel(data.llmModel || 'Llama3');
      setTemperature(
        typeof data.temperature === 'number' ? data.temperature : 0.7,
      );
      setStream(data.stream !== false);
    } catch {
      // ignore
    }
  }, []);

  const handleSave = () => {
    localStorage.setItem(
      'bioflow_settings',
      JSON.stringify({
        molEncoder,
        protEncoder,
        bindingPredictor,
        propertyPredictor,
        llmProvider,
        llmModel,
        temperature,
        stream,
      }),
    );
    setSaveStatus('Saved');
    setTimeout(() => setSaveStatus(null), 2000);
  };

  return (
    <div className="animate-in fade-in space-y-8 duration-500">
      <PageHeader
        title="Settings"
        subtitle="Configure models, databases, and preferences"
        icon={<Settings className="text-primary h-8 w-8" />}
      />

      <Tabs defaultValue="models" className="w-full">
        <TabsList className="w-full justify-start overflow-x-auto">
          <TabsTrigger value="models">Models</TabsTrigger>
          <TabsTrigger value="database">Database</TabsTrigger>
          <TabsTrigger value="api">Endpoints</TabsTrigger>
          <TabsTrigger value="appearance">Appearance</TabsTrigger>
          <TabsTrigger value="system">System</TabsTrigger>
        </TabsList>
        <TabsContent value="models" className="mt-6 space-y-6">
          <SectionHeader
            title="Model Configuration"
            icon={<Brain className="text-primary h-5 w-5" />}
          />

          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Embedding Models</CardTitle>
                <CardDescription>
                  Configure models used for molecular and protein embeddings
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="mol-encoder">Molecule Encoder</Label>
                  <Select value={molEncoder} onValueChange={setMolEncoder}>
                    <SelectTrigger id="mol-encoder">
                      <SelectValue placeholder="Select molecule encoder" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="MolCLR">
                        MolCLR (Recommended)
                      </SelectItem>
                      <SelectItem value="ChemBERTa">ChemBERTa</SelectItem>
                      <SelectItem value="GraphMVP">GraphMVP</SelectItem>
                      <SelectItem value="MolBERT">MolBERT</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="prot-encoder">Protein Encoder</Label>
                  <Select value={protEncoder} onValueChange={setProtEncoder}>
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
                <CardDescription>
                  Configure downstream task predictors
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="binding">Binding Predictor</Label>
                  <Select
                    value={bindingPredictor}
                    onValueChange={setBindingPredictor}
                  >
                    <SelectTrigger id="binding">
                      <SelectValue placeholder="Select predictor" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="DrugBAN">
                        DrugBAN (Recommended)
                      </SelectItem>
                      <SelectItem value="DeepDTA">DeepDTA</SelectItem>
                      <SelectItem value="GraphDTA">GraphDTA</SelectItem>
                      <SelectItem value="Custom">Custom</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="property">Property Predictor</Label>
                  <Select
                    value={propertyPredictor}
                    onValueChange={setPropertyPredictor}
                  >
                    <SelectTrigger id="property">
                      <SelectValue placeholder="Select predictor" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ADMET-AI">
                        ADMET-AI (Recommended)
                      </SelectItem>
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
              <CardDescription>
                Configure open-source local models for evidence retrieval and
                reasoning
              </CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="llm-provider">LLM Provider</Label>
                <Select value={llmProvider} onValueChange={setLlmProvider}>
                  <SelectTrigger id="llm-provider">
                    <SelectValue placeholder="Select provider" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Local">Local (Open-Source)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="llm-model">Model</Label>
                <Select value={llmModel} onValueChange={setLlmModel}>
                  <SelectTrigger id="llm-model">
                    <SelectValue placeholder="Select model" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Llama3">Llama 3</SelectItem>
                    <SelectItem value="Mistral">Mistral</SelectItem>
                    <SelectItem value="Qwen">Qwen</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="col-span-1 space-y-4 md:col-span-2">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Temperature: {temperature.toFixed(1)}</Label>
                    <span className="text-muted-foreground text-xs">
                      Creativity vs Precision
                    </span>
                  </div>
                  <Slider
                    value={[temperature]}
                    max={1}
                    step={0.1}
                    onValueChange={([v]) =>
                      v !== undefined && setTemperature(v)
                    }
                  />
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="stream"
                    checked={stream}
                    onCheckedChange={setStream}
                  />
                  <Label htmlFor="stream">Stream Responses</Label>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="appearance">
          <Card>
            <CardContent className="text-muted-foreground p-12 text-center">
              Theme settings coming soon.
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="database">
          <Card>
            <CardContent className="text-muted-foreground p-12 text-center">
              Database connection settings.
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="api">
          <Card>
            <CardContent className="text-muted-foreground p-12 text-center">
              Configure local model endpoints or proxies.
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <div className="fixed right-6 bottom-6 flex items-center gap-3">
        {saveStatus && (
          <span className="text-sm text-emerald-500">{saveStatus}</span>
        )}
        <Button size="lg" className="shadow-2xl" onClick={handleSave}>
          <Save className="mr-2 h-4 w-4" />
          Save Changes
        </Button>
      </div>
    </div>
  );
}
