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

interface Variant {
  rank: number;
  id: string;
  content: string;
  modality: string;
  similarity_score: number;
  diversity_score: number;
  justification: string;
  evidence_links: Array<{ source: string; identifier: string; url: string }>;
  metadata: Record<string, unknown>;
}

interface DesignAssistantModalProps {
  isOpen: boolean;
  onClose: () => void;
  sourceItem: string; // SMILES string or protein sequence
  sourceModality?: 'molecule' | 'protein' | 'auto';
  sourceName?: string;
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

                            {/* Justification */}
                            <p className="text-sm text-muted-foreground leading-relaxed">
                              {variant.justification}
                            </p>

                            {/* Evidence links */}
                            {variant.evidence_links &&
                              variant.evidence_links.length > 0 && (
                                <div className="flex flex-wrap gap-2 mt-2">
                                  {variant.evidence_links.map((link, linkIdx) => (
                                    <a
                                      key={linkIdx}
                                      href={link.url}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="inline-flex items-center gap-1 text-xs text-blue-500 hover:text-blue-600 hover:underline"
                                    >
                                      <ExternalLink className="h-3 w-3" />
                                      {link.source}: {link.identifier}
                                    </a>
                                  ))}
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

                          {/* Scores column */}
                          <div className="flex flex-col items-end gap-1 shrink-0">
                            <div className="text-right">
                              <div className="text-xs text-muted-foreground">
                                Diversity
                              </div>
                              <div
                                className={`text-sm font-semibold ${
                                  variant.diversity_score >= 0.5
                                    ? 'text-green-600'
                                    : 'text-amber-600'
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
              generated
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
