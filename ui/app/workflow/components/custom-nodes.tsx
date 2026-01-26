'use client';

import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Database, FlaskConical, FileSearch, Shield, Brain, Dna, FileText } from 'lucide-react';

// Base node wrapper with consistent styling
const NodeWrapper = ({ 
  children, 
  color, 
  label,
  icon: Icon,
  status = 'idle'
}: { 
  children?: React.ReactNode; 
  color: string;
  label: string;
  icon: React.ElementType;
  status?: 'idle' | 'running' | 'complete' | 'error';
}) => {
  const statusColors = {
    idle: 'border-gray-600',
    running: 'border-yellow-500 animate-pulse',
    complete: 'border-green-500',
    error: 'border-red-500'
  };

  return (
    <div className={`
      min-w-[180px] rounded-xl border-2 ${statusColors[status]}
      bg-gray-900/95 backdrop-blur shadow-2xl overflow-hidden
    `}>
      <div className={`px-3 py-2 ${color} flex items-center gap-2`}>
        <Icon className="w-4 h-4 text-white" />
        <span className="text-sm font-semibold text-white">{label}</span>
      </div>
      <div className="p-3 text-xs text-gray-300">
        {children}
      </div>
    </div>
  );
};

// Data Input Node
export const DataInputNode = memo(({ data }: NodeProps) => (
  <>
    <NodeWrapper color="bg-blue-600" label="Data Input" icon={FileText} status={data.status}>
      <div className="space-y-1">
        <div className="text-gray-400">Type: <span className="text-white">{data.inputType || 'SMILES'}</span></div>
        <div className="text-gray-400 truncate max-w-[150px]">
          Input: <span className="text-white font-mono">{data.input || 'CC(=O)O...'}</span>
        </div>
      </div>
    </NodeWrapper>
    <Handle type="source" position={Position.Right} className="w-3 h-3 !bg-blue-500" />
  </>
));
DataInputNode.displayName = 'DataInputNode';

// DeepPurpose Generator Node
export const GeneratorNode = memo(({ data }: NodeProps) => (
  <>
    <Handle type="target" position={Position.Left} className="w-3 h-3 !bg-purple-500" />
    <NodeWrapper color="bg-purple-600" label="DeepPurpose DTI" icon={FlaskConical} status={data.status}>
      <div className="space-y-1">
        <div className="text-gray-400">Model: <span className="text-white">Morgan + CNN</span></div>
        <div className="text-gray-400">Encoding: <span className="text-green-400">256D</span></div>
        {data.prediction && (
          <div className="text-gray-400">Affinity: <span className="text-emerald-400">{data.prediction}</span></div>
        )}
      </div>
    </NodeWrapper>
    <Handle type="source" position={Position.Right} className="w-3 h-3 !bg-purple-500" />
  </>
));
GeneratorNode.displayName = 'GeneratorNode';

// Qdrant Storage Node
export const QdrantNode = memo(({ data }: NodeProps) => (
  <>
    <Handle type="target" position={Position.Left} className="w-3 h-3 !bg-orange-500" />
    <NodeWrapper color="bg-orange-600" label="Qdrant Vector DB" icon={Database} status={data.status}>
      <div className="space-y-1">
        <div className="text-gray-400">Collection: <span className="text-white">bio_discovery</span></div>
        <div className="text-gray-400">Vectors: <span className="text-cyan-400">{data.vectorCount || '23,531'}</span></div>
        <div className="text-gray-400">Index: <span className="text-white">HNSW</span></div>
      </div>
    </NodeWrapper>
    <Handle type="source" position={Position.Right} className="w-3 h-3 !bg-orange-500" />
  </>
));
QdrantNode.displayName = 'QdrantNode';

// Similarity Search Node
export const SearchNode = memo(({ data }: NodeProps) => (
  <>
    <Handle type="target" position={Position.Left} className="w-3 h-3 !bg-cyan-500" />
    <NodeWrapper color="bg-cyan-600" label="Similarity Search" icon={FileSearch} status={data.status}>
      <div className="space-y-1">
        <div className="text-gray-400">Top-K: <span className="text-white">{data.topK || 10}</span></div>
        <div className="text-gray-400">Metric: <span className="text-white">Cosine</span></div>
        {data.results && (
          <div className="text-gray-400">Found: <span className="text-green-400">{data.results} matches</span></div>
        )}
      </div>
    </NodeWrapper>
    <Handle type="source" position={Position.Right} className="w-3 h-3 !bg-cyan-500" />
  </>
));
SearchNode.displayName = 'SearchNode';

// Validator Node
export const ValidatorNode = memo(({ data }: NodeProps) => (
  <>
    <Handle type="target" position={Position.Left} className="w-3 h-3 !bg-green-500" />
    <NodeWrapper color="bg-green-600" label="Validator Agent" icon={Shield} status={data.status}>
      <div className="space-y-1">
        <div className="text-gray-400">Toxicity: <span className={data.toxicity === 'Low' ? 'text-green-400' : 'text-red-400'}>{data.toxicity || 'Checking...'}</span></div>
        <div className="text-gray-400">Novelty: <span className="text-white">{data.novelty || 'Pending'}</span></div>
        <div className="text-gray-400">Score: <span className="text-yellow-400">{data.score || '—'}</span></div>
      </div>
    </NodeWrapper>
    <Handle type="source" position={Position.Right} className="w-3 h-3 !bg-green-500" />
  </>
));
ValidatorNode.displayName = 'ValidatorNode';

// OpenBioMed Multimodal Node
export const MultimodalNode = memo(({ data }: NodeProps) => (
  <>
    <Handle type="target" position={Position.Left} className="w-3 h-3 !bg-pink-500" />
    <NodeWrapper color="bg-pink-600" label="OpenBioMed" icon={Brain} status={data.status}>
      <div className="space-y-1">
        <div className="text-gray-400">Mode: <span className="text-white">{data.mode || 'Cross-Modal'}</span></div>
        <div className="text-gray-400">Modalities: <span className="text-white">Protein + Text</span></div>
        <div className="text-gray-400">Embeddings: <span className="text-pink-400">768D</span></div>
      </div>
    </NodeWrapper>
    <Handle type="source" position={Position.Right} className="w-3 h-3 !bg-pink-500" />
  </>
));
MultimodalNode.displayName = 'MultimodalNode';

// Output/Results Node
export const OutputNode = memo(({ data }: NodeProps) => (
  <>
    <Handle type="target" position={Position.Left} className="w-3 h-3 !bg-emerald-500" />
    <NodeWrapper color="bg-emerald-600" label="Results" icon={Dna} status={data.status}>
      <div className="space-y-1">
        <div className="text-gray-400">Candidates: <span className="text-white">{data.candidates || 0}</span></div>
        <div className="text-gray-400">Top Match: <span className="text-emerald-400">{data.topMatch || '—'}</span></div>
        <div className="text-gray-400">Confidence: <span className="text-yellow-400">{data.confidence || '—'}</span></div>
      </div>
    </NodeWrapper>
  </>
));
OutputNode.displayName = 'OutputNode';

export const nodeTypes = {
  dataInput: DataInputNode,
  generator: GeneratorNode,
  qdrant: QdrantNode,
  search: SearchNode,
  validator: ValidatorNode,
  multimodal: MultimodalNode,
  output: OutputNode,
};
