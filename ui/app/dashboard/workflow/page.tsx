'use client';

import {
  ArrowRight,
  Check,
  ChevronDown,
  ChevronRight,
  Download,
  Loader2,
  Play,
  Plus,
  Settings,
  Sparkles,
  Trash2,
  Upload,
} from 'lucide-react';
import * as React from 'react';

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
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { runWorkflow as apiRunWorkflow } from '@/lib/api';
import { WorkflowResult, WorkflowStep } from '@/schemas/workflow';

function GenerateStepConfig({
  config,
  onChange,
}: {
  config: Record<string, unknown>;
  onChange: (config: Record<string, unknown>) => void;
}) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label>Mode</Label>
        <Select
          value={(config.mode as string) || 'text'}
          onValueChange={(v) => onChange({ ...config, mode: v })}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="text">Text-to-Molecule</SelectItem>
            <SelectItem value="mutate">Mutation-Based</SelectItem>
            <SelectItem value="scaffold">Scaffold-Based</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label>Prompt / SMILES</Label>
        <Textarea
          placeholder={
            config.mode === 'text'
              ? 'Describe the molecule you want to generate...'
              : config.mode === 'mutate'
                ? 'Enter a SMILES string to mutate...'
                : 'Enter a scaffold SMILES...'
          }
          value={(config.prompt as string) || ''}
          onChange={(e) => onChange({ ...config, prompt: e.target.value })}
          className="h-24"
        />
      </div>

      <div className="space-y-2">
        <Label>Number to Generate: {Number(config.num_candidates) || 5}</Label>
        <Slider
          value={[Number(config['num_candidates']) || 5]}
          onValueChange={([v]) =>
            v !== undefined && onChange({ ...config, num_candidates: v })
          }
          min={1}
          max={20}
          step={1}
        />
      </div>
    </div>
  );
}

function ValidateStepConfig({
  config,
  onChange,
}: {
  config: Record<string, unknown>;
  onChange: (config: Record<string, unknown>) => void;
}) {
  const checks = (config.checks as string[]) || [
    'lipinski',
    'admet',
    'qed',
    'alerts',
  ];

  const toggleCheck = (check: string) => {
    const newChecks = checks.includes(check)
      ? checks.filter((c) => c !== check)
      : [...checks, check];
    onChange({ ...config, checks: newChecks });
  };

  return (
    <div className="space-y-4">
      <Label>Validation Checks</Label>
      <div className="grid grid-cols-2 gap-2">
        {[
          { id: 'lipinski', label: 'Lipinski Rule of 5' },
          { id: 'admet', label: 'ADMET Properties' },
          { id: 'qed', label: 'QED Score' },
          { id: 'alerts', label: 'Structural Alerts' },
        ].map((check) => (
          <div
            key={check.id}
            className={`flex cursor-pointer items-center gap-2 rounded-lg border p-2 transition-colors ${checks.includes(check.id)
              ? 'border-primary bg-primary/10'
              : 'border-border hover:border-primary/50'
              }`}
            onClick={() => toggleCheck(check.id)}
          >
            <div
              className={`flex h-4 w-4 items-center justify-center rounded border ${checks.includes(check.id)
                ? 'bg-primary border-primary'
                : 'border-muted-foreground'
                }`}
            >
              {checks.includes(check.id) && (
                <Check className="text-primary-foreground h-3 w-3" />
              )}
            </div>
            <span className="text-sm">{check.label}</span>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-between">
        <Label>Strict Mode</Label>
        <Switch
          checked={Boolean(config.strict)}
          onCheckedChange={(v) => onChange({ ...config, strict: v })}
        />
      </div>
    </div>
  );
}

function RankStepConfig({
  config,
  onChange,
}: {
  config: Record<string, unknown>;
  onChange: (config: Record<string, unknown>) => void;
}) {
  const weights = (config.weights as Record<string, number>) || {
    qed: 0.3,
    validity: 0.3,
    mw: 0.2,
    logp: 0.2,
  };

  const updateWeight = (key: string, value: number) => {
    onChange({ ...config, weights: { ...weights, [key]: value } });
  };

  return (
    <div className="space-y-4">
      <Label>Ranking Weights</Label>

      {Object.entries({
        qed: 'QED Score',
        validity: 'Validity',
        mw: 'Molecular Weight',
        logp: 'LogP',
      }).map(([key, label]) => (
        <div key={key} className="space-y-1">
          <div className="flex justify-between text-sm">
            <span>{label}</span>
            <span className="text-muted-foreground">
              {(weights[key] || 0).toFixed(2)}
            </span>
          </div>
          <Slider
            value={[weights[key] || 0]}
            onValueChange={([v]) => v !== undefined && updateWeight(key, v)}
            min={0}
            max={1}
            step={0.05}
          />
        </div>
      ))}

      <div className="space-y-2">
        <Label>Top K: {Number(config.top_k) || 5}</Label>
        <Slider
          value={[Number(config.top_k) || 5]}
          onValueChange={([v]) => onChange({ ...config, top_k: v })}
          min={1}
          max={20}
          step={1}
        />
      </div>
    </div>
  );
}

function StepCard({
  step,
  index,
  onUpdate,
  onRemove,
  isLast,
}: {
  step: WorkflowStep;
  index: number;
  onUpdate: (step: WorkflowStep) => void;
  onRemove: () => void;
  isLast: boolean;
}) {
  const [isExpanded, setIsExpanded] = React.useState(true);

  const getStepIcon = () => {
    switch (step.type) {
      case 'generate':
        return <Sparkles className="h-4 w-4" />;
      case 'validate':
        return <Check className="h-4 w-4" />;
      case 'rank':
        return <Settings className="h-4 w-4" />;
    }
  };

  const getStatusBadge = () => {
    switch (step.status) {
      case 'pending':
        return <Badge variant="outline">Pending</Badge>;
      case 'running':
        return <Badge className="bg-blue-500">Running</Badge>;
      case 'completed':
        return <Badge className="bg-green-500">Completed</Badge>;
      case 'error':
        return <Badge variant="destructive">Error</Badge>;
    }
  };

  return (
    <div className="relative">
      <Card
        className={step.status === 'running' ? 'border-blue-500 shadow-lg' : ''}
      >
        <CardHeader className="py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-primary/10 text-primary flex h-8 w-8 items-center justify-center rounded-full">
                {index + 1}
              </div>
              <div className="flex items-center gap-2">
                {getStepIcon()}
                <span className="font-medium">{step.name}</span>
              </div>
              {getStatusBadge()}
            </div>
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setIsExpanded(!isExpanded)}
              >
                {isExpanded ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={onRemove}
                className="text-destructive hover:text-destructive"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>

        {isExpanded && (
          <CardContent className="pt-0">
            {step.type === 'generate' && (
              <GenerateStepConfig
                config={step.config}
                onChange={(config) => onUpdate({ ...step, config })}
              />
            )}
            {step.type === 'validate' && (
              <ValidateStepConfig
                config={step.config}
                onChange={(config) => onUpdate({ ...step, config })}
              />
            )}
            {step.type === 'rank' && (
              <RankStepConfig
                config={step.config}
                onChange={(config) => onUpdate({ ...step, config })}
              />
            )}

            {step.error && (
              <div className="bg-destructive/10 text-destructive mt-4 rounded-lg p-3 text-sm">
                {step.error}
              </div>
            )}
          </CardContent>
        )}
      </Card>

      {!isLast && (
        <div className="flex justify-center py-2">
          <ArrowRight className="text-muted-foreground h-6 w-6" />
        </div>
      )}
    </div>
  );
}

function WorkflowResults({ result }: { result: WorkflowResult | null }) {
  if (!result) return null;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Results</CardTitle>
          <Badge variant="outline">{result.total_time_ms.toFixed(0)}ms</Badge>
        </div>
        <CardDescription>
          {result.candidates.length} candidates generated •{' '}
          {result.steps_completed} steps completed
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[300px]">
          <div className="space-y-3">
            {result.candidates.map((candidate, idx) => (
              <div key={idx} className="bg-card rounded-lg border p-3">
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{candidate.name}</span>
                      <Badge
                        variant={
                          candidate.validation.is_valid
                            ? 'default'
                            : 'destructive'
                        }
                      >
                        {candidate.validation.is_valid ? 'Valid' : 'Invalid'}
                      </Badge>
                    </div>
                    <code className="text-muted-foreground block max-w-md truncate text-xs">
                      {candidate.smiles}
                    </code>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold">
                      {candidate.score.toFixed(3)}
                    </div>
                    <span className="text-muted-foreground text-xs">Score</span>
                  </div>
                </div>

                <div className="text-muted-foreground mt-2 flex gap-4 text-xs">
                  {Object.entries(candidate.validation.properties).map(
                    ([key, value]) => (
                      <span key={key}>
                        {key}:{' '}
                        {typeof value === 'number' ? value.toFixed(2) : value}
                      </span>
                    ),
                  )}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

export default function WorkflowBuilderPage() {
  const [steps, setSteps] = React.useState<WorkflowStep[]>([
    {
      id: 'gen-1',
      type: 'generate',
      name: 'Generate Molecules',
      config: { mode: 'text', prompt: '', num_candidates: 5 },
      status: 'pending',
    },
    {
      id: 'val-1',
      type: 'validate',
      name: 'Validate Candidates',
      config: { checks: ['lipinski', 'admet', 'qed', 'alerts'], strict: false },
      status: 'pending',
    },
    {
      id: 'rank-1',
      type: 'rank',
      name: 'Rank & Select',
      config: {
        weights: { qed: 0.3, validity: 0.3, mw: 0.2, logp: 0.2 },
        top_k: 5,
      },
      status: 'pending',
    },
  ]);
  const [isRunning, setIsRunning] = React.useState(false);
  const [progress, setProgress] = React.useState(0);
  const [result, setResult] = React.useState<WorkflowResult | null>(null);
  const [workflowName, setWorkflowName] = React.useState(
    'My Discovery Workflow',
  );

  const templates = [
    {
      name: 'Standard Discovery',
      description: 'Generate, validate, then rank candidates.',
      steps: [
        {
          type: 'generate',
          name: 'Generate Molecules',
          config: { mode: 'text', prompt: '', num_candidates: 8 },
        },
        {
          type: 'validate',
          name: 'Validate Candidates',
          config: {
            checks: ['lipinski', 'admet', 'qed', 'alerts'],
            strict: false,
          },
        },
        {
          type: 'rank',
          name: 'Rank & Select',
          config: {
            weights: { qed: 0.3, validity: 0.3, mw: 0.2, logp: 0.2 },
            top_k: 5,
          },
        },
      ],
    },
    {
      name: 'Scaffold Expansion',
      description: 'Generate scaffold variants and rank by QED.',
      steps: [
        {
          type: 'generate',
          name: 'Scaffold Variants',
          config: { mode: 'scaffold', prompt: '', num_candidates: 6 },
        },
        {
          type: 'rank',
          name: 'Rank & Select',
          config: { weights: { qed: 0.6, validity: 0.4 }, top_k: 5 },
        },
      ],
    },
    {
      name: 'Fast Validation',
      description: 'Validate existing candidates quickly.',
      steps: [
        {
          type: 'validate',
          name: 'Validate Candidates',
          config: { checks: ['lipinski', 'qed'], strict: false },
        },
      ],
    },
  ];

  const applyTemplate = (template: (typeof templates)[number]) => {
    setWorkflowName(template.name);
    setSteps(
      template.steps.map((s, idx) => ({
        id: `${s.type}-${idx}`,
        type: s.type as WorkflowStep['type'],
        name: s.name,
        config: s.config,
        status: 'pending',
      })),
    );
  };

  const addStep = (type: WorkflowStep['type']) => {
    const id = `${type}-${Date.now()}`;
    const newStep: WorkflowStep = {
      id,
      type,
      name:
        type === 'generate'
          ? 'Generate Molecules'
          : type === 'validate'
            ? 'Validate Candidates'
            : 'Rank & Select',
      config:
        type === 'generate'
          ? { mode: 'text', prompt: '', num_candidates: 5 }
          : type === 'validate'
            ? { checks: ['lipinski', 'admet', 'qed', 'alerts'], strict: false }
            : {
              weights: { qed: 0.3, validity: 0.3, mw: 0.2, logp: 0.2 },
              top_k: 5,
            },
      status: 'pending',
    };
    setSteps([...steps, newStep]);
  };

  const updateStep = (updatedStep: WorkflowStep) => {
    setSteps(steps.map((s) => (s.id === updatedStep.id ? updatedStep : s)));
  };

  const removeStep = (id: string) => {
    setSteps(steps.filter((s) => s.id !== id));
  };

  const runWorkflow = async () => {
    setIsRunning(true);
    setProgress(0);
    setResult(null);

    setSteps(
      steps.map((s) => ({
        ...s,
        status: 'pending',
        result: undefined,
        error: undefined,
      })),
    );

    try {
      const generateStep = steps.find((s) => s.type === 'generate');
      const validateStep = steps.find((s) => s.type === 'validate');
      const rankStep = steps.find((s) => s.type === 'rank');

      if (!generateStep) {
        throw new Error('Workflow must include a generate step');
      }

      setSteps((prev) =>
        prev.map((s) =>
          s.id === generateStep.id ? { ...s, status: 'running' } : s,
        ),
      );
      setProgress(10);

      const data = await apiRunWorkflow({
        query: (generateStep.config.prompt as string) || 'drug-like molecule',
        num_candidates: (generateStep.config.num_candidates as number) || 5,
        top_k: (rankStep?.config.top_k as number) || 5,
      });

      setProgress(30);
      setSteps((prev) =>
        prev.map((s) =>
          s.id === generateStep.id
            ? { ...s, status: 'completed' }
            : s.type === 'validate'
              ? { ...s, status: 'running' }
              : s,
        ),
      );

      await new Promise((r) => setTimeout(r, 500));
      setProgress(60);

      if (validateStep) {
        setSteps((prev) =>
          prev.map((s) =>
            s.id === validateStep.id
              ? { ...s, status: 'completed' }
              : s.type === 'rank'
                ? { ...s, status: 'running' }
                : s,
          ),
        );
      }

      setSteps((prev) => prev.map((s) => ({ ...s, status: 'completed' })));
      setProgress(100);

      const workflowResult: WorkflowResult = {
        candidates: (data.top_candidates || []).map((c) => ({
          smiles: c.smiles || '',
          name: c.name || `Candidate ${c.rank || 0}`,
          validation: c.validation || {
            is_valid: true,
            checks: {},
            properties: {},
          },
          score: c.score || 0,
        })),
        steps_completed: data.steps_completed || steps.length,
        total_time_ms: data.execution_time_ms || 0,
      };

      setResult(workflowResult);
    } catch (err) {
      console.error('Workflow error:', err);
      setSteps((prev) =>
        prev.map((s) =>
          s.status === 'running'
            ? { ...s, status: 'error', error: String(err) }
            : s,
        ),
      );
    } finally {
      setIsRunning(false);
    }
  };

  const exportWorkflow = () => {
    const config = {
      name: workflowName,
      steps: steps.map((s) => ({
        type: s.type,
        name: s.name,
        config: s.config,
      })),
    };
    const json = JSON.stringify(config, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${workflowName.replace(/\s+/g, '_').toLowerCase()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const importWorkflow = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;

      const text = await file.text();
      try {
        const config = JSON.parse(text);
        setWorkflowName(config.name || 'Imported Workflow');
        setSteps(
          config.steps.map((s: Record<string, unknown>, idx: number) => ({
            id: `${s.type}-${idx}`,
            type: s.type,
            name: s.name,
            config: s.config,
            status: 'pending',
          })),
        );
      } catch (err) {
        console.error('Failed to import workflow:', err);
      }
    };
    input.click();
  };

  return (
    <div className="container mx-auto space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h1 className="text-3xl font-bold tracking-tight">
            Workflow Builder
          </h1>
          <p className="text-muted-foreground">
            Design and execute drug discovery pipelines with visual
            configuration
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={importWorkflow}>
            <Upload className="mr-2 h-4 w-4" />
            Import
          </Button>
          <Button variant="outline" onClick={exportWorkflow}>
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
          <Button
            onClick={runWorkflow}
            disabled={isRunning || steps.length === 0}
          >
            {isRunning ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Play className="mr-2 h-4 w-4" />
            )}
            Run Workflow
          </Button>
        </div>
      </div>

      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center gap-4">
            <Label className="whitespace-nowrap">Workflow Name:</Label>
            <Input
              value={workflowName}
              onChange={(e) => setWorkflowName(e.target.value)}
              className="max-w-md"
            />
          </div>
        </CardContent>
      </Card>

      <div id="templates">
        <Card>
          <CardHeader>
            <CardTitle>Templates</CardTitle>
            <CardDescription>
              Load a preset workflow configuration.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {templates.map((template) => (
              <button
                key={template.name}
                onClick={() => applyTemplate(template)}
                className="hover:bg-accent rounded-lg border p-4 text-left transition-colors"
              >
                <div className="font-semibold">{template.name}</div>
                <div className="text-muted-foreground text-sm">
                  {template.description}
                </div>
              </button>
            ))}
          </CardContent>
        </Card>
      </div>

      {isRunning && (
        <Card>
          <CardContent className="pt-6">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Running workflow...</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} />
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Pipeline Steps</CardTitle>
              <CardDescription>
                Configure each step of your discovery workflow
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {steps.length === 0 ? (
                <div className="text-muted-foreground py-8 text-center">
                  No steps added yet. Click &quot;Add Step&quot; to get started.
                </div>
              ) : (
                steps.map((step, idx) => (
                  <StepCard
                    key={step.id}
                    step={step}
                    index={idx}
                    onUpdate={updateStep}
                    onRemove={() => removeStep(step.id)}
                    isLast={idx === steps.length - 1}
                  />
                ))
              )}

              <div className="flex gap-2 border-t pt-4">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => addStep('generate')}
                >
                  <Plus className="mr-1 h-4 w-4" />
                  Add Generate
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => addStep('validate')}
                >
                  <Plus className="mr-1 h-4 w-4" />
                  Add Validate
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => addStep('rank')}
                >
                  <Plus className="mr-1 h-4 w-4" />
                  Add Rank
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-1">
          <WorkflowResults result={result} />

          {!result && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Results</CardTitle>
                <CardDescription>
                  Run the workflow to see results
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-muted-foreground py-8 text-center">
                  Configure your pipeline and click &quot;Run Workflow&quot; to execute
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
