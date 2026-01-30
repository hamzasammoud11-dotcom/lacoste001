'use client';

import * as React from 'react';
import {
  AlertCircle,
  Copy,
  ExternalLink,
  Loader2,
  Sparkles,
  TrendingUp,
  Beaker,
  Dna,
  FileText,
  FlaskConical,
  BookOpen,
  Star,
  Lightbulb,
} from 'lucide-react';

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogOverlay,
  DialogPortal,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { getDesignVariants } from '@/lib/api';

interface EvidenceLink {
  source: string;
  identifier: string;
  url: string;
  label?: string;
}

interface JustificationPart {
  type: 'priority' | 'rationale' | 'evidence';
  text: string;
  icon?: string;
}

interface Variant {
  rank: number;
  id: string;
  content: string;
  modality: string;
  similarity_score: number;
  priority_score?: number;  // Priority ≠ Similarity - based on evidence strength
  diversity_score: number;
  justification: string;
  evidence_links: EvidenceLink[];
  metadata: Record<string, unknown>;
}

interface DesignAssistantModalProps {
  isOpen: boolean;
  onClose: () => void;
  sourceItem: string; // SMILES string or protein sequence
  sourceModality?: 'molecule' | 'protein' | 'auto';
  sourceName?: string;
}

// Parse justification to extract structured parts (top-level helper)
function parseJustificationText(justification: string): JustificationPart[] {
  const parts: JustificationPart[] = [];
  let text = justification;
  
  // Check for priority indicator
  if (text.startsWith('⭐ HIGH PRIORITY:')) {
    parts.push({ type: 'priority', text: 'HIGH PRIORITY', icon: '⭐' });
    text = text.replace('⭐ HIGH PRIORITY: ', '');
  } else if (text.startsWith('PROMISING:')) {
    parts.push({ type: 'priority', text: 'PROMISING', icon: '💡' });
    text = text.replace('PROMISING: ', '');
  }
  
  // Split by separators
  const segments = text.split(' | ');
  segments.forEach(segment => {
    if (segment.includes('📄') || segment.includes('PMID')) {
      parts.push({ type: 'evidence', text: segment.replace('📄 ', ''), icon: '📄' });
    } else if (segment.includes('📊') || segment.includes('📝') || segment.includes('🧪')) {
      const icon = segment.match(/^[📊📝🧪]/)?.[0] || '📊';
      parts.push({ type: 'evidence', text: segment.replace(/^[📊📝🧪] /, ''), icon });
    } else if (segment.startsWith('**')) {
      // Extract bolded hypothesis
      const match = segment.match(/\*\*([^*]+)\*\*:?\s*(.*)/);
      if (match) {
        parts.push({ type: 'rationale', text: `${match[1]}: ${match[2]}` });
      } else {
        parts.push({ type: 'rationale', text: segment.replace(/\*\*/g, '') });
      }
    } else if (segment.trim()) {
      parts.push({ type: 'rationale', text: segment });
    }
  });
  
  return parts;
}

export function DesignAssistantModal({
  isOpen,
  onClose,
  sourceItem,
  sourceModality = 'auto',
  sourceName,
}: DesignAssistantModalProps) {
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [variants, setVariants] = React.useState<Variant[]>([]);
  const [referenceModality, setReferenceModality] = React.useState<string>('');

  // Fetch variants when modal opens
  React.useEffect(() => {
    if (isOpen && sourceItem) {
      fetchVariants();
    }
  }, [isOpen, sourceItem]);

  const fetchVariants = async () => {
    setIsLoading(true);
    setError(null);
    setVariants([]);

    try {
      const response = await getDesignVariants({
        reference: sourceItem,
        modality: sourceModality,
        num_variants: 5,
        diversity: 0.5,
      });

      setVariants(response.variants || []);
      setReferenceModality(response.reference_modality || sourceModality);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to generate variants'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const getModalityIcon = (modality: string) => {
    if (modality === 'molecule' || modality === 'drug') {
      return <Beaker className="h-4 w-4" />;
    }
    if (modality === 'protein' || modality === 'target') {
      return <Dna className="h-4 w-4" />;
    }
    return <Sparkles className="h-4 w-4" />;
  };

  const formatScore = (score: number) => {
    return (score * 100).toFixed(1) + '%';
  };

  const highlightDifferences = (original: string, variant: string) => {
    // Simple diff highlight - shows what's different
    if (original.length < 20 && variant.length < 20) {
      // For short SMILES, just show them
      return variant;
    }
    // For longer sequences, show truncated with note
    return variant.length > 60 ? variant.slice(0, 60) + '...' : variant;
  };

  // Get source icon based on evidence source type
  const getSourceIcon = (source: string): React.ReactNode => {
    const sourceIcons: Record<string, React.ReactNode> = {
      pubmed: <BookOpen className="h-3 w-3" />,
      chembl: <FlaskConical className="h-3 w-3" />,
      uniprot: <Dna className="h-3 w-3" />,
      pdb: <Dna className="h-3 w-3" />,
      experiment: <Beaker className="h-3 w-3" />,
      doi: <FileText className="h-3 w-3" />,
    };
    return sourceIcons[source.toLowerCase()] || <ExternalLink className="h-3 w-3" />;
  };

  // Get human-readable source label
  const getSourceLabel = (source: string, identifier: string) => {
    const labels: Record<string, string> = {
      pubmed: `PubMed: ${identifier}`,
      chembl_compound: `ChEMBL: ${identifier}`,
      chembl_target: `Target: ${identifier}`,
      uniprot: `UniProt: ${identifier}`,
      pdb: `PDB: ${identifier}`,
      doi: `DOI: ${identifier}`,
      experiment: `Experiment: ${identifier}`,
    };
    return labels[source.toLowerCase()] || `${source}: ${identifier}`;
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogPortal>
        <DialogOverlay />
        <DialogContent className="max-w-2xl max-h-[85vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-purple-500" />
              Design Variants
            </DialogTitle>
            <DialogDescription>
              AI-suggested alternatives for{' '}
              <span className="font-mono text-foreground">
                {sourceName || sourceItem.slice(0, 30)}
                {sourceItem.length > 30 ? '...' : ''}
              </span>
            </DialogDescription>
          </DialogHeader>

          <div className="flex-1 overflow-hidden">
            {isLoading ? (
              <div className="flex flex-col items-center justify-center py-12 gap-3">
                <Loader2 className="h-8 w-8 animate-spin text-purple-500" />
                <p className="text-sm text-muted-foreground">
                  Generating design variants...
                </p>
                <p className="text-xs text-muted-foreground">
                  Analyzing structural features and exploring chemical space
                </p>
              </div>
            ) : error ? (
              <div className="flex flex-col items-center justify-center py-8 gap-3">
                <AlertCircle className="h-8 w-8 text-destructive" />
                <p className="text-sm text-destructive font-medium">
                  Generation Failed
                </p>
                <p className="text-xs text-muted-foreground text-center max-w-md">
                  {error}
                </p>
                <Button variant="outline" size="sm" onClick={fetchVariants}>
                  Try Again
                </Button>
              </div>
            ) : variants.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 gap-3">
                <Sparkles className="h-8 w-8 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  No variants found
                </p>
                <p className="text-xs text-muted-foreground text-center">
                  Try a different source compound or protein sequence
                </p>
              </div>
            ) : (
              <ScrollArea className="h-[400px] pr-4">
                <div className="space-y-3">
                  {/* Reference summary */}
                  <div className="bg-muted/50 rounded-lg p-3 mb-4">
                    <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                      {getModalityIcon(referenceModality)}
                      <span>Reference ({referenceModality})</span>
                    </div>
                    <code className="text-xs font-mono break-all">
                      {sourceItem.slice(0, 80)}
                      {sourceItem.length > 80 ? '...' : ''}
                    </code>
                  </div>

                  {/* Variants list */}
                  {variants.map((variant, idx) => (
                    <Card
                      key={variant.id || idx}
                      className="border-l-4 border-l-purple-500/50 hover:border-l-purple-500 transition-colors"
                    >
                      <CardContent className="p-4">
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex-1 min-w-0">
                            {/* Header with rank and badges */}
                            <div className="flex items-center gap-2 mb-2 flex-wrap">
                              <Badge
                                variant="outline"
                                className="bg-purple-500/10 text-purple-600 border-purple-500/30"
                              >
                                #{variant.rank}
                              </Badge>
                              <Badge variant="secondary" className="text-xs">
                                {getModalityIcon(variant.modality)}
                                <span className="ml-1">{variant.modality}</span>
                              </Badge>
                              {variant.similarity_score > 0 && (
                                <Badge
                                  variant="outline"
                                  className={`text-xs ${
                                    variant.similarity_score >= 0.8
                                      ? 'bg-green-500/10 text-green-600 border-green-500/30'
                                      : variant.similarity_score >= 0.6
                                        ? 'bg-amber-500/10 text-amber-600 border-amber-500/30'
                                        : 'bg-gray-500/10'
                                  }`}
                                >
                                  <TrendingUp className="h-3 w-3 mr-1" />
                                  {formatScore(variant.similarity_score)} similar
                                </Badge>
                              )}
                            </div>

                            {/* Variant content */}
                            <div className="bg-muted/30 rounded p-2 mb-2">
                              <code className="text-xs font-mono break-all text-foreground">
                                {highlightDifferences(
                                  sourceItem,
                                  variant.content
                                )}
                              </code>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 px-2 ml-2"
                                onClick={() => copyToClipboard(variant.content)}
                              >
                                <Copy className="h-3 w-3" />
                              </Button>
                            </div>

                            {/* Justification - Now with rich formatting */}
                            <VariantJustification parts={parseJustificationText(variant.justification)} />

                            {/* Evidence links - Enhanced display */}
                            <VariantEvidenceLinks links={variant.evidence_links} getSourceIcon={getSourceIcon} getSourceLabel={getSourceLabel} />

                            {/* Lab notes / Abstract snippet if available */}
                            {Boolean(variant.metadata?.notes) && (
                              <div className="bg-amber-50 dark:bg-amber-950/30 rounded-md p-2 mt-2">
                                <div className="text-xs font-medium text-amber-700 dark:text-amber-300 flex items-center gap-1 mb-1">
                                  📝 Lab Notes
                                </div>
                                <p className="text-xs text-amber-600 dark:text-amber-400 italic">
                                  "{String(variant.metadata.notes).slice(0, 200)}
                                  {String(variant.metadata.notes).length > 200 ? '...' : ''}"
                                </p>
                              </div>
                            )}

                            {Boolean(variant.metadata?.abstract) && (
                              <div className="bg-green-50 dark:bg-green-950/30 rounded-md p-2 mt-2">
                                <div className="text-xs font-medium text-green-700 dark:text-green-300 flex items-center gap-1 mb-1">
                                  📄 Abstract Excerpt
                                </div>
                                <p className="text-xs text-green-600 dark:text-green-400 italic">
                                  "{String(variant.metadata.abstract).slice(0, 200)}
                                  {String(variant.metadata.abstract).length > 200 ? '...' : ''}"
                                </p>
                              </div>
                            )}

                            {Boolean(variant.metadata?.protocol) && (
                              <div className="bg-purple-50 dark:bg-purple-950/30 rounded-md p-2 mt-2">
                                <div className="text-xs font-medium text-purple-700 dark:text-purple-300 flex items-center gap-1 mb-1">
                                  🧪 Protocol
                                </div>
                                <p className="text-xs text-purple-600 dark:text-purple-400">
                                  {String(variant.metadata.protocol)}
                                </p>
                              </div>
                            )}

                            {/* Metadata badges */}
                            {variant.metadata && Object.keys(variant.metadata).length > 0 && (
                              <div className="flex flex-wrap gap-1 mt-2">
                                {typeof variant.metadata.name === 'string' && variant.metadata.name && (
                                  <Badge variant="outline" className="text-xs">
                                    {variant.metadata.name}
                                  </Badge>
                                )}
                                {typeof variant.metadata.affinity_class === 'string' && variant.metadata.affinity_class && (
                                  <Badge variant="outline" className="text-xs">
                                    {variant.metadata.affinity_class}
                                  </Badge>
                                )}
                              </div>
                            )}
                          </div>

                          {/* Scores column - PRIORITY vs SIMILARITY distinction */}
                          <div className="flex flex-col items-end gap-2 shrink-0 min-w-[100px]">
                            {/* Priority Score - This is the RANKING signal */}
                            <div className="text-right">
                              <div className="text-xs font-medium text-purple-600 dark:text-purple-400 flex items-center gap-1 justify-end">
                                <Star className="h-3 w-3" />
                                Priority
                              </div>
                              <div
                                className={`text-lg font-bold ${
                                  (variant.priority_score || variant.metadata?.priority_score as number || variant.similarity_score * 0.8) >= 0.7
                                    ? 'text-purple-600'
                                    : (variant.priority_score || variant.metadata?.priority_score as number || variant.similarity_score * 0.8) >= 0.5
                                      ? 'text-amber-600'
                                      : 'text-gray-500'
                                }`}
                              >
                                {formatScore(variant.priority_score || variant.metadata?.priority_score as number || variant.similarity_score * 0.8)}
                              </div>
                            </div>
                            
                            {/* Similarity Score - The RAW vector distance */}
                            <div className="text-right border-t pt-1">
                              <div className="text-[10px] text-muted-foreground">
                                Similarity
                              </div>
                              <div className="text-xs text-muted-foreground">
                                {formatScore(variant.similarity_score)}
                              </div>
                            </div>
                            
                            {/* Diversity Score */}
                            <div className="text-right">
                              <div className="text-[10px] text-muted-foreground">
                                Diversity
                              </div>
                              <div
                                className={`text-xs ${
                                  variant.diversity_score >= 0.4
                                    ? 'text-green-600'
                                    : 'text-muted-foreground'
                                }`}
                              >
                                {formatScore(variant.diversity_score)}
                              </div>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </ScrollArea>
            )}
          </div>

          <div className="flex justify-between items-center pt-4 border-t mt-4">
            <p className="text-xs text-muted-foreground">
              {variants.length} variant{variants.length !== 1 ? 's' : ''}{' '}
              generated • Sorted by <span className="font-medium text-purple-600">Priority</span> (not just similarity)
            </p>
            <div className="flex gap-2">
              <Button variant="outline" onClick={fetchVariants} disabled={isLoading}>
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                ) : (
                  <Sparkles className="h-4 w-4 mr-2" />
                )}
                Regenerate
              </Button>
              <Button onClick={onClose}>Close</Button>
            </div>
          </div>
        </DialogContent>
      </DialogPortal>
    </Dialog>
  );
}

// Helper component for rendering variant justification
function VariantJustification({ parts }: { parts: JustificationPart[] }) {
  return (
    <div className="space-y-2">
      {/* Priority badge */}
      {parts.some(p => p.type === 'priority') && (
        <div className="flex items-center gap-1">
          {parts.filter(p => p.type === 'priority').map((p, pidx) => (
            <Badge 
              key={pidx}
              variant="default" 
              className={`text-xs ${
                p.text === 'HIGH PRIORITY' 
                  ? 'bg-amber-500 hover:bg-amber-600' 
                  : 'bg-blue-500 hover:bg-blue-600'
              }`}
            >
              <Star className="h-3 w-3 mr-1" />
              {p.text}
            </Badge>
          ))}
        </div>
      )}
      
      {/* Design rationale */}
      {parts.filter(p => p.type === 'rationale').map((p, pidx) => (
        <div key={pidx} className="flex items-start gap-2 text-sm">
          <Lightbulb className="h-4 w-4 mt-0.5 text-amber-500 shrink-0" />
          <span className="text-muted-foreground leading-relaxed">
            {p.text}
          </span>
        </div>
      ))}
      
      {/* Evidence snippets */}
      {parts.filter(p => p.type === 'evidence').length > 0 && (
        <div className="bg-blue-50 dark:bg-blue-950/30 rounded-md p-2 space-y-1">
          <div className="text-xs font-medium text-blue-700 dark:text-blue-300 flex items-center gap-1">
            <FileText className="h-3 w-3" />
            Evidence
          </div>
          {parts.filter(p => p.type === 'evidence').map((p, pidx) => (
            <div key={pidx} className="text-xs text-blue-600 dark:text-blue-400 pl-4">
              {p.icon && <span className="mr-1">{p.icon}</span>}
              {p.text}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Helper component for rendering evidence links
function VariantEvidenceLinks({ 
  links, 
  getSourceIcon, 
  getSourceLabel 
}: { 
  links: EvidenceLink[]; 
  getSourceIcon: (source: string) => React.ReactNode;
  getSourceLabel: (source: string, identifier: string) => string;
}) {
  if (!links || links.length === 0) {
    return null;
  }
  
  return (
    <div className="border-t pt-2 mt-2">
      <div className="text-xs font-medium text-muted-foreground mb-1 flex items-center gap-1">
        <BookOpen className="h-3 w-3" />
        Source References ({links.length})
      </div>
      <div className="flex flex-wrap gap-2">
        {links.map((link, linkIdx) => (
          <a
            key={linkIdx}
            href={link.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-xs bg-muted hover:bg-muted/80 px-2 py-1 rounded-md text-blue-600 dark:text-blue-400 hover:text-blue-700 transition-colors"
          >
            {getSourceIcon(link.source)}
            <span>{getSourceLabel(link.source, link.identifier)}</span>
            <ExternalLink className="h-2.5 w-2.5 opacity-50" />
          </a>
        ))}
      </div>
    </div>
  );
}
