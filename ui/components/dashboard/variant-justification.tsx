'use client';

import {
  AlertTriangle,
  ArrowRight,
  Beaker,
  BookOpen,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  Dna,
  ExternalLink,
  FlaskConical,
  HelpCircle,
  Lightbulb,
  ShieldCheck,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
  Zap,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';

// Simple tooltip replacement (tooltip component not available)
const TooltipProvider = ({ children }: { children: React.ReactNode }) => <>{children}</>;
const Tooltip = ({ children }: { children: React.ReactNode }) => <span className="relative group">{children}</span>;
const TooltipTrigger = ({ children, asChild }: { children: React.ReactNode; asChild?: boolean }) => <span>{children}</span>;
const TooltipContent = ({ children }: { children: React.ReactNode }) => (
  <span className="hidden group-hover:block absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 text-xs bg-popover text-popover-foreground rounded shadow-lg z-50 whitespace-nowrap">
    {children}
  </span>
);

// Types for variant justification
export interface ChemicalModification {
  position: string; // e.g., "R1", "Position 4", "N-terminus"
  originalGroup: string;
  newGroup: string;
  effect: string; // e.g., "improves membrane permeability"
  confidence: number; // 0-1
}

export interface SimilarCompound {
  id: string;
  name: string;
  smiles?: string;
  experimentId?: string;
  outcome: 'success' | 'partial' | 'failure';
  activity?: string; // e.g., "IC50: 45 nM"
  url?: string;
}

export interface PredictedProperty {
  name: string;
  value: number | string;
  unit?: string;
  comparison?: {
    baseline: number | string;
    improvement: number; // percentage change
    direction: 'better' | 'worse' | 'neutral';
  };
  confidence?: number;
}

export interface EvidenceSource {
  type: 'paper' | 'experiment' | 'database' | 'prediction';
  id: string;
  title: string;
  url?: string;
  relevance: number;
}

export interface VariantJustificationData {
  variantId: string;
  summary: string;
  reasoning: string[];
  modifications: ChemicalModification[];
  similarCompounds: SimilarCompound[];
  predictedProperties: PredictedProperty[];
  evidenceSources: EvidenceSource[];
  overallConfidence: number;
  riskFactors?: string[];
  suggestedExperiments?: string[];
}

// Evidence strength configurations
const CONFIDENCE_CONFIG = {
  high: {
    icon: ShieldCheck,
    color: 'text-green-600 dark:text-green-400',
    bgColor: 'bg-green-500/10',
    borderColor: 'border-green-500/30',
    label: 'High Confidence',
  },
  medium: {
    icon: CheckCircle2,
    color: 'text-blue-600 dark:text-blue-400',
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-blue-500/30',
    label: 'Medium Confidence',
  },
  low: {
    icon: AlertTriangle,
    color: 'text-amber-600 dark:text-amber-400',
    bgColor: 'bg-amber-500/10',
    borderColor: 'border-amber-500/30',
    label: 'Low Confidence',
  },
  unknown: {
    icon: HelpCircle,
    color: 'text-gray-500 dark:text-gray-400',
    bgColor: 'bg-gray-500/10',
    borderColor: 'border-gray-500/30',
    label: 'Unknown',
  },
};

function getConfidenceLevel(score: number): keyof typeof CONFIDENCE_CONFIG {
  if (score >= 0.7) return 'high';
  if (score >= 0.4) return 'medium';
  if (score > 0) return 'low';
  return 'unknown';
}

// Main Justification Card Component
interface VariantJustificationCardProps {
  data: VariantJustificationData;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
  className?: string;
}

export function VariantJustificationCard({
  data,
  isExpanded = false,
  onToggleExpand,
  className,
}: VariantJustificationCardProps) {
  const confidenceLevel = getConfidenceLevel(data.overallConfidence);
  const config = CONFIDENCE_CONFIG[confidenceLevel];
  const ConfidenceIcon = config.icon;

  return (
    <Card className={`border-l-4 ${config.borderColor} ${className}`}>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="h-4 w-4 text-amber-500" />
              <span className="font-semibold text-sm">Why This Variant?</span>
              <Badge className={`${config.bgColor} ${config.color} border-0`}>
                <ConfidenceIcon className="h-3 w-3 mr-1" />
                {config.label}
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground">{data.summary}</p>
          </div>
          
          {onToggleExpand && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7"
              onClick={onToggleExpand}
            >
              {isExpanded ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
          )}
        </div>
      </CardHeader>
      
      <CardContent className="pt-0">
        {/* Quick summary - always visible */}
        <div className="space-y-3">
          {/* Key modifications */}
          {data.modifications.length > 0 && (
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground flex items-center gap-1">
                <Zap className="h-3 w-3" />
                Key Modifications
              </div>
              <div className="flex flex-wrap gap-1">
                {data.modifications.slice(0, isExpanded ? undefined : 2).map((mod, idx) => (
                  <ModificationBadge key={idx} modification={mod} />
                ))}
                {!isExpanded && data.modifications.length > 2 && (
                  <Badge variant="outline" className="text-xs">
                    +{data.modifications.length - 2} more
                  </Badge>
                )}
              </div>
            </div>
          )}
          
          {/* Predicted improvements - always visible */}
          {data.predictedProperties.length > 0 && (
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground flex items-center gap-1">
                <Target className="h-3 w-3" />
                Predicted Improvements
              </div>
              <div className="grid grid-cols-2 gap-2">
                {data.predictedProperties.slice(0, isExpanded ? undefined : 2).map((prop, idx) => (
                  <PredictedPropertyCard key={idx} property={prop} compact />
                ))}
              </div>
            </div>
          )}
        </div>
        
        {/* Expanded content */}
        {isExpanded && (
          <div className="space-y-4 mt-4 pt-4 border-t">
            {/* Reasoning steps */}
            {data.reasoning.length > 0 && (
              <div className="space-y-2">
                <div className="text-xs font-medium text-muted-foreground">
                  Design Rationale
                </div>
                <div className="space-y-1">
                  {data.reasoning.map((reason, idx) => (
                    <div key={idx} className="flex items-start gap-2 text-sm">
                      <div className="flex-shrink-0 w-5 h-5 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-medium">
                        {idx + 1}
                      </div>
                      <span className="text-muted-foreground">{reason}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {/* Similar compounds */}
            {data.similarCompounds.length > 0 && (
              <SimilarCompoundsSection compounds={data.similarCompounds} />
            )}
            
            {/* Evidence sources */}
            {data.evidenceSources.length > 0 && (
              <EvidenceSourcesSection sources={data.evidenceSources} />
            )}
            
            {/* Risk factors */}
            {data.riskFactors && data.riskFactors.length > 0 && (
              <div className="space-y-2">
                <div className="text-xs font-medium text-amber-600 flex items-center gap-1">
                  <AlertTriangle className="h-3 w-3" />
                  Potential Risks
                </div>
                <ul className="text-xs text-muted-foreground space-y-1">
                  {data.riskFactors.map((risk, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-amber-500">•</span>
                      {risk}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {/* Suggested experiments */}
            {data.suggestedExperiments && data.suggestedExperiments.length > 0 && (
              <div className="space-y-2">
                <div className="text-xs font-medium text-green-600 flex items-center gap-1">
                  <Beaker className="h-3 w-3" />
                  Suggested Validation Experiments
                </div>
                <ul className="text-xs text-muted-foreground space-y-1">
                  {data.suggestedExperiments.map((exp, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-green-500">→</span>
                      {exp}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Modification badge component
function ModificationBadge({ modification }: { modification: ChemicalModification }) {
  const confidenceLevel = getConfidenceLevel(modification.confidence);
  const config = CONFIDENCE_CONFIG[confidenceLevel];

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger>
          <Badge
            variant="outline"
            className={`text-xs cursor-help ${config.borderColor}`}
          >
            <span className="font-mono mr-1">{modification.position}:</span>
            <span className="text-muted-foreground line-through mr-1">
              {modification.originalGroup}
            </span>
            <ArrowRight className="h-3 w-3 mx-0.5" />
            <span className={config.color}>{modification.newGroup}</span>
          </Badge>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs">
          <p className="text-xs font-medium mb-1">Effect: {modification.effect}</p>
          <p className="text-xs text-muted-foreground">
            Confidence: {(modification.confidence * 100).toFixed(0)}%
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// Predicted property card
function PredictedPropertyCard({
  property,
  compact = false,
}: {
  property: PredictedProperty;
  compact?: boolean;
}) {
  const hasComparison = property.comparison !== undefined;
  const improvementPercent = property.comparison?.improvement || 0;
  const direction = property.comparison?.direction || 'neutral';

  const directionConfig = {
    better: { icon: TrendingUp, color: 'text-green-600', bgColor: 'bg-green-500/10' },
    worse: { icon: TrendingDown, color: 'text-red-600', bgColor: 'bg-red-500/10' },
    neutral: { icon: Sparkles, color: 'text-gray-500', bgColor: 'bg-gray-500/10' },
  };

  const config = directionConfig[direction];
  const DirectionIcon = config.icon;

  if (compact) {
    return (
      <div className={`rounded-md p-2 ${config.bgColor}`}>
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">{property.name}</span>
          {hasComparison && (
            <div className={`flex items-center gap-0.5 text-xs ${config.color}`}>
              <DirectionIcon className="h-3 w-3" />
              {Math.abs(improvementPercent) > 0 && (
                <span>{improvementPercent > 0 ? '+' : ''}{improvementPercent}%</span>
              )}
            </div>
          )}
        </div>
        <div className="font-semibold text-sm">
          {property.value} {property.unit}
        </div>
        {hasComparison && (
          <div className="text-xs text-muted-foreground">
            vs {property.comparison?.baseline} {property.unit}
          </div>
        )}
      </div>
    );
  }

  return (
    <Card className="p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium">{property.name}</span>
        {property.confidence !== undefined && (
          <Badge variant="secondary" className="text-[10px] h-4 px-1">
            {(property.confidence * 100).toFixed(0)}% conf.
          </Badge>
        )}
      </div>
      
      <div className="flex items-end justify-between">
        <div>
          <div className="text-2xl font-bold">
            {property.value}
            {property.unit && <span className="text-sm font-normal ml-1">{property.unit}</span>}
          </div>
          {hasComparison && (
            <div className="text-xs text-muted-foreground mt-1">
              Parent: {property.comparison?.baseline} {property.unit}
            </div>
          )}
        </div>
        
        {hasComparison && (
          <div className={`flex items-center gap-1 px-2 py-1 rounded ${config.bgColor} ${config.color}`}>
            <DirectionIcon className="h-4 w-4" />
            <span className="text-sm font-medium">
              {improvementPercent > 0 ? '+' : ''}{improvementPercent}%
            </span>
          </div>
        )}
      </div>
    </Card>
  );
}

// Similar compounds section
function SimilarCompoundsSection({ compounds }: { compounds: SimilarCompound[] }) {
  const outcomeConfig = {
    success: { color: 'text-green-600', bgColor: 'bg-green-500/10', label: 'Success' },
    partial: { color: 'text-amber-600', bgColor: 'bg-amber-500/10', label: 'Partial' },
    failure: { color: 'text-red-600', bgColor: 'bg-red-500/10', label: 'Failed' },
  };

  return (
    <div className="space-y-2">
      <div className="text-xs font-medium text-muted-foreground flex items-center gap-1">
        <FlaskConical className="h-3 w-3" />
        Similar Compounds with Known Results
      </div>
      <div className="space-y-1">
        {compounds.map((compound, idx) => {
          const config = outcomeConfig[compound.outcome];
          return (
            <div
              key={idx}
              className="flex items-center justify-between p-2 rounded-md bg-muted/30 hover:bg-muted/50 transition-colors"
            >
              <div className="flex items-center gap-2">
                <Badge variant="outline" className={`text-[10px] ${config.bgColor} ${config.color} border-0`}>
                  {config.label}
                </Badge>
                <span className="text-xs font-medium">{compound.name}</span>
                {compound.experimentId && (
                  <span className="text-xs text-muted-foreground">
                    ({compound.experimentId})
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                {compound.activity && (
                  <span className="text-xs font-mono text-muted-foreground">
                    {compound.activity}
                  </span>
                )}
                {compound.url && (
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-5 w-5"
                    onClick={() => window.open(compound.url, '_blank')}
                  >
                    <ExternalLink className="h-3 w-3" />
                  </Button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Evidence sources section
function EvidenceSourcesSection({ sources }: { sources: EvidenceSource[] }) {
  const typeConfig = {
    paper: { icon: BookOpen, color: 'text-emerald-600' },
    experiment: { icon: Beaker, color: 'text-purple-600' },
    database: { icon: Dna, color: 'text-cyan-600' },
    prediction: { icon: Sparkles, color: 'text-amber-600' },
  };

  return (
    <div className="space-y-2">
      <div className="text-xs font-medium text-muted-foreground flex items-center gap-1">
        <BookOpen className="h-3 w-3" />
        Supporting Evidence
      </div>
      <div className="space-y-1">
        {sources.map((source, idx) => {
          const config = typeConfig[source.type];
          const Icon = config.icon;
          return (
            <div
              key={idx}
              className="flex items-center justify-between p-2 rounded-md bg-muted/30"
            >
              <div className="flex items-center gap-2 flex-1 min-w-0">
                <Icon className={`h-3 w-3 ${config.color} shrink-0`} />
                <span className="text-xs truncate">{source.title}</span>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <Progress value={source.relevance * 100} className="w-12 h-1" />
                <span className="text-[10px] text-muted-foreground w-8">
                  {(source.relevance * 100).toFixed(0)}%
                </span>
                {source.url && (
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-5 w-5"
                    onClick={() => window.open(source.url, '_blank')}
                  >
                    <ExternalLink className="h-3 w-3" />
                  </Button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Inline justification for design modal (compact version)
interface InlineVariantJustificationProps {
  summary: string;
  modifications?: ChemicalModification[];
  predictedIC50?: { value: number; unit: string; parent?: number };
  similarSuccess?: { name: string; experimentId: string };
  confidence: number;
}

export function InlineVariantJustification({
  summary,
  modifications = [],
  predictedIC50,
  similarSuccess,
  confidence,
}: InlineVariantJustificationProps) {
  const confidenceLevel = getConfidenceLevel(confidence);
  const config = CONFIDENCE_CONFIG[confidenceLevel];

  return (
    <div className="space-y-2 text-sm">
      {/* Main summary */}
      <div className="flex items-start gap-2">
        <Lightbulb className="h-4 w-4 mt-0.5 text-amber-500 shrink-0" />
        <span className="text-muted-foreground">{summary}</span>
      </div>
      
      {/* Modifications */}
      {modifications.length > 0 && (
        <div className="bg-muted/30 rounded-md p-2 space-y-1">
          {modifications.map((mod, idx) => (
            <div key={idx} className="flex items-center gap-2 text-xs">
              <span className="text-muted-foreground">•</span>
              <span className="font-mono font-medium">{mod.position}:</span>
              <span className="text-muted-foreground">
                {mod.originalGroup} → <span className="text-primary">{mod.newGroup}</span>
              </span>
              <span className="text-muted-foreground italic">({mod.effect})</span>
            </div>
          ))}
        </div>
      )}
      
      {/* Similar success reference */}
      {similarSuccess && (
        <div className="flex items-center gap-2 text-xs text-green-600 bg-green-500/10 rounded-md px-2 py-1">
          <CheckCircle2 className="h-3 w-3" />
          <span>
            Similar to <span className="font-medium">{similarSuccess.name}</span> which succeeded in{' '}
            <span className="font-mono">{similarSuccess.experimentId}</span>
          </span>
        </div>
      )}
      
      {/* Predicted IC50 */}
      {predictedIC50 && (
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1">
            <Target className="h-3 w-3 text-primary" />
            <span className="text-muted-foreground">Predicted IC50:</span>
            <span className="font-mono font-medium">
              {predictedIC50.value} {predictedIC50.unit}
            </span>
          </div>
          {predictedIC50.parent && (
            <div className="flex items-center gap-1 text-green-600">
              <TrendingUp className="h-3 w-3" />
              <span>
                vs parent {predictedIC50.parent} {predictedIC50.unit} (
                {Math.round(((predictedIC50.parent - predictedIC50.value) / predictedIC50.parent) * 100)}% better)
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Export types for use in other components
// Types are already exported with their interfaces
