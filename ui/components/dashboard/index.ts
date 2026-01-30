// Dashboard components index
// Export all components for easy importing

// Evidence Chain Visualization
export {
  EvidenceChainVisualization,
  EvidenceChainModal,
  EvidenceChainLegend,
  type EvidenceNode,
  type EvidenceEdge,
  type EvidenceChain,
} from './evidence-chain';

// Enhanced Faceted Filtering
export {
  EnhancedFacetedFilter,
  filtersToQuery,
  EXPERIMENT_FILTER_FIELDS,
  COMPOUND_FILTER_FIELDS,
  type FilterCondition,
  type FilterGroup,
  type FilterConfig,
  type FilterFieldConfig,
} from './enhanced-filters';

// Explore From Here Navigation
export {
  ExploreDropdown,
  ExplorePanel,
  ExploreModal,
  ExploreContextMenu,
  useExploreData,
  type RelatedItem,
  type ExploreCategory,
  type ExploreFromHereData,
} from './explore-from-here';

// Variant Justification
export {
  VariantJustificationCard,
  InlineVariantJustification,
  type VariantJustificationData,
  type ChemicalModification,
  type SimilarCompound,
  type PredictedProperty,
  type EvidenceSource,
} from './variant-justification';

// Design Assistant (existing)
export { DesignAssistantModal } from './design/design-assistant-modal';
