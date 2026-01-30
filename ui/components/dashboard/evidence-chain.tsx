'use client';

import * as React from 'react';
import {
  ArrowRight,
  Beaker,
  BookOpen,
  ChevronDown,
  ChevronRight,
  Dna,
  ExternalLink,
  FlaskConical,
  GitBranch,
  Loader2,
  Microscope,
  Network,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogPortal,
  DialogOverlay,
} from '@/components/ui/dialog';

// Types for evidence chain nodes
export interface EvidenceNode {
  id: string;
  type: 'compound' | 'experiment' | 'paper' | 'protein' | 'image';
  label: string;
  subtitle?: string;
  score?: number;
  url?: string;
  metadata?: Record<string, unknown>;
}

export interface EvidenceEdge {
  from: string;
  to: string;
  relationship: string;
  strength?: number;
}

export interface EvidenceChain {
  nodes: EvidenceNode[];
  edges: EvidenceEdge[];
  rootId: string;
}

interface EvidenceChainProps {
  chain: EvidenceChain | null;
  isLoading?: boolean;
  onNodeClick?: (node: EvidenceNode) => void;
  onExplore?: (nodeId: string, nodeType: string) => void;
  className?: string;
}

// Node type configurations
const NODE_CONFIG: Record<EvidenceNode['type'], {
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  bgColor: string;
  borderColor: string;
}> = {
  compound: {
    icon: FlaskConical,
    color: 'text-blue-600 dark:text-blue-400',
    bgColor: 'bg-blue-50 dark:bg-blue-950/30',
    borderColor: 'border-blue-200 dark:border-blue-800',
  },
  experiment: {
    icon: Beaker,
    color: 'text-purple-600 dark:text-purple-400',
    bgColor: 'bg-purple-50 dark:bg-purple-950/30',
    borderColor: 'border-purple-200 dark:border-purple-800',
  },
  paper: {
    icon: BookOpen,
    color: 'text-emerald-600 dark:text-emerald-400',
    bgColor: 'bg-emerald-50 dark:bg-emerald-950/30',
    borderColor: 'border-emerald-200 dark:border-emerald-800',
  },
  protein: {
    icon: Dna,
    color: 'text-cyan-600 dark:text-cyan-400',
    bgColor: 'bg-cyan-50 dark:bg-cyan-950/30',
    borderColor: 'border-cyan-200 dark:border-cyan-800',
  },
  image: {
    icon: Microscope,
    color: 'text-amber-600 dark:text-amber-400',
    bgColor: 'bg-amber-50 dark:bg-amber-950/30',
    borderColor: 'border-amber-200 dark:border-amber-800',
  },
};

// Single node component
function EvidenceNodeCard({
  node,
  isRoot,
  isExpanded,
  onToggle,
  onClick,
  onExplore,
}: {
  node: EvidenceNode;
  isRoot?: boolean;
  isExpanded?: boolean;
  onToggle?: () => void;
  onClick?: () => void;
  onExplore?: () => void;
}) {
  const config = NODE_CONFIG[node.type];
  const Icon = config.icon;

  return (
    <div
      className={`
        relative flex items-start gap-3 p-3 rounded-lg border transition-all cursor-pointer
        ${config.bgColor} ${config.borderColor}
        ${isRoot ? 'ring-2 ring-primary ring-offset-2' : ''}
        hover:shadow-md
      `}
      onClick={onClick}
    >
      <div className={`flex-shrink-0 p-2 rounded-lg ${config.bgColor}`}>
        <Icon className={`h-5 w-5 ${config.color}`} />
      </div>
      
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm truncate">{node.label}</span>
          {isRoot && (
            <Badge variant="outline" className="text-[10px] h-4 px-1">Root</Badge>
          )}
          {node.score !== undefined && (
            <Badge variant="secondary" className="text-[10px] h-4 px-1">
              {(node.score * 100).toFixed(0)}%
            </Badge>
          )}
        </div>
        
        {node.subtitle && (
          <p className="text-xs text-muted-foreground truncate mt-0.5">
            {node.subtitle}
          </p>
        )}
        
        <div className="flex items-center gap-1 mt-2">
          {onToggle && (
            <Button
              size="sm"
              variant="ghost"
              className="h-6 px-2 text-xs"
              onClick={(e) => { e.stopPropagation(); onToggle(); }}
            >
              {isExpanded ? (
                <ChevronDown className="h-3 w-3 mr-1" />
              ) : (
                <ChevronRight className="h-3 w-3 mr-1" />
              )}
              Details
            </Button>
          )}
          
          {onExplore && (
            <Button
              size="sm"
              variant="ghost"
              className="h-6 px-2 text-xs"
              onClick={(e) => { e.stopPropagation(); onExplore(); }}
            >
              <Network className="h-3 w-3 mr-1" />
              Explore
            </Button>
          )}
          
          {node.url && (
            <Button
              size="sm"
              variant="ghost"
              className="h-6 px-2 text-xs"
              onClick={(e) => { e.stopPropagation(); window.open(node.url, '_blank'); }}
            >
              <ExternalLink className="h-3 w-3 mr-1" />
              Open
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

// Connection arrow component
function ConnectionArrow({ relationship, strength }: { relationship: string; strength?: number }) {
  return (
    <div className="flex items-center gap-2 py-1 pl-8">
      <div className="flex items-center text-muted-foreground">
        <div className="w-4 border-t border-dashed border-muted-foreground/50" />
        <ArrowRight className="h-3 w-3" />
      </div>
      <span className="text-xs text-muted-foreground italic">
        {relationship}
        {strength !== undefined && (
          <span className="ml-1 text-[10px]">({(strength * 100).toFixed(0)}%)</span>
        )}
      </span>
    </div>
  );
}

// Main evidence chain visualization component
export function EvidenceChainVisualization({
  chain,
  isLoading,
  onNodeClick,
  onExplore,
  className,
}: EvidenceChainProps) {
  const [expandedNodes, setExpandedNodes] = React.useState<Set<string>>(new Set());

  if (isLoading) {
    return (
      <Card className={className}>
        <CardContent className="py-8">
          <div className="flex flex-col items-center justify-center gap-3">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            <p className="text-sm text-muted-foreground">Building evidence chain...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!chain || chain.nodes.length === 0) {
    return (
      <Card className={className}>
        <CardContent className="py-8">
          <div className="flex flex-col items-center justify-center gap-2 text-center">
            <GitBranch className="h-8 w-8 text-muted-foreground/50" />
            <p className="text-sm text-muted-foreground">No evidence chain available</p>
            <p className="text-xs text-muted-foreground">
              Select a result to see its evidence connections
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Build a tree structure from the chain
  const nodeMap = new Map(chain.nodes.map(n => [n.id, n]));
  const childrenMap = new Map<string, { node: EvidenceNode; edge: EvidenceEdge }[]>();
  
  chain.edges.forEach(edge => {
    const children = childrenMap.get(edge.from) || [];
    const targetNode = nodeMap.get(edge.to);
    if (targetNode) {
      children.push({ node: targetNode, edge });
      childrenMap.set(edge.from, children);
    }
  });

  // Recursive render function
  const renderNode = (nodeId: string, depth: number = 0): React.ReactNode => {
    const node = nodeMap.get(nodeId);
    if (!node) return null;

    const children = childrenMap.get(nodeId) || [];
    const isRoot = nodeId === chain.rootId;
    const isExpanded = expandedNodes.has(nodeId);

    return (
      <div key={nodeId} className="space-y-1" style={{ marginLeft: depth > 0 ? '1.5rem' : 0 }}>
        <EvidenceNodeCard
          node={node}
          isRoot={isRoot}
          isExpanded={isExpanded}
          onToggle={children.length > 0 ? () => {
            setExpandedNodes(prev => {
              const next = new Set(prev);
              if (next.has(nodeId)) next.delete(nodeId);
              else next.add(nodeId);
              return next;
            });
          } : undefined}
          onClick={() => onNodeClick?.(node)}
          onExplore={() => onExplore?.(node.id, node.type)}
        />
        
        {(isExpanded || depth === 0) && children.length > 0 && (
          <div className="space-y-1">
            {children.map(({ node: childNode, edge }) => (
              <React.Fragment key={childNode.id}>
                <ConnectionArrow relationship={edge.relationship} strength={edge.strength} />
                {renderNode(childNode.id, depth + 1)}
              </React.Fragment>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <GitBranch className="h-4 w-4 text-primary" />
          Evidence Chain
          <Badge variant="secondary" className="text-xs">
            {chain.nodes.length} nodes
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="max-h-[400px] pr-3">
          <div className="space-y-2">
            {renderNode(chain.rootId)}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

// Modal version for detailed evidence chain view
interface EvidenceChainModalProps {
  isOpen: boolean;
  onClose: () => void;
  chain: EvidenceChain | null;
  isLoading?: boolean;
  title?: string;
  onNodeClick?: (node: EvidenceNode) => void;
  onExplore?: (nodeId: string, nodeType: string) => void;
}

export function EvidenceChainModal({
  isOpen,
  onClose,
  chain,
  isLoading,
  title,
  onNodeClick,
  onExplore,
}: EvidenceChainModalProps) {
  const handleClose = () => {
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && handleClose()}>
      <DialogPortal>
        <DialogOverlay />
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <GitBranch className="h-5 w-5 text-primary" />
              {title || 'Evidence Chain'}
            </DialogTitle>
            <DialogDescription>
              Trace the connections between compounds, experiments, and publications
            </DialogDescription>
          </DialogHeader>
          
          <div className="flex-1 overflow-hidden">
            <EvidenceChainVisualization
              chain={chain}
              isLoading={isLoading}
              onNodeClick={onNodeClick}
              onExplore={onExplore}
            />
          </div>
          
          <div className="flex justify-end pt-4 border-t mt-4">
            <Button onClick={handleClose}>Close</Button>
          </div>
        </DialogContent>
      </DialogPortal>
    </Dialog>
  );
}

// Legend component for understanding node types
export function EvidenceChainLegend() {
  return (
    <div className="flex flex-wrap items-center gap-3 text-xs">
      {Object.entries(NODE_CONFIG).map(([type, config]) => {
        const Icon = config.icon;
        return (
          <div key={type} className="flex items-center gap-1">
            <Icon className={`h-3 w-3 ${config.color}`} />
            <span className="capitalize text-muted-foreground">{type}</span>
          </div>
        );
      })}
    </div>
  );
}
