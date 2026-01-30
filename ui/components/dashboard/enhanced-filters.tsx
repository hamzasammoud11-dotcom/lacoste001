'use client';

import * as React from 'react';
import {
  ChevronDown,
  Filter,
  Plus,
  Trash2,
  X,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

// Simple checkbox replacement (checkbox component not available)
const Checkbox = ({ 
  checked, 
  onCheckedChange, 
  id 
}: { 
  checked: boolean; 
  onCheckedChange: (checked: boolean) => void; 
  id: string;
}) => (
  <input 
    type="checkbox" 
    id={id} 
    checked={checked} 
    onChange={(e) => onCheckedChange(e.target.checked)}
    className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
  />
);

// Simple label
const Label = ({ 
  htmlFor, 
  children,
  className = ''
}: { 
  htmlFor?: string; 
  children: React.ReactNode;
  className?: string;
}) => (
  <label htmlFor={htmlFor} className={`text-sm font-medium ${className}`}>
    {children}
  </label>
);

// Simple popover replacement with click-outside handling
const Popover = ({ children, open, onOpenChange }: { children: React.ReactNode; open?: boolean; onOpenChange?: (open: boolean) => void }) => {
  const [internalOpen, setInternalOpen] = React.useState(false);
  const isOpen = open !== undefined ? open : internalOpen;
  const setIsOpen = onOpenChange || setInternalOpen;
  const popoverRef = React.useRef<HTMLDivElement>(null);

  // Handle click outside to close
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    
    if (isOpen) {
      // Delay to prevent immediate close on open click
      setTimeout(() => {
        document.addEventListener('mousedown', handleClickOutside);
      }, 0);
    }
    
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen, setIsOpen]);

  return (
    <div className="relative" ref={popoverRef}>
      {React.Children.map(children, child => {
        if (React.isValidElement(child)) {
          if (child.type === PopoverTrigger) {
            return React.cloneElement(child as React.ReactElement<{ onClick: () => void }>, {
              onClick: () => setIsOpen(!isOpen)
            });
          }
          if (child.type === PopoverContent && isOpen) {
            return React.cloneElement(child as React.ReactElement<{ onClose: () => void }>, {
              onClose: () => setIsOpen(false)
            });
          }
        }
        return child;
      })}
    </div>
  );
};
const PopoverTrigger = ({ children, asChild: _asChild, onClick }: { children: React.ReactNode; asChild?: boolean; onClick?: () => void }) => (
  <span onClick={onClick} className="cursor-pointer">{children}</span>
);
const PopoverContent = ({ children, className = '', onClose: _onClose }: { children: React.ReactNode; className?: string; onClose?: () => void }) => (
  <div className={`absolute top-full left-0 mt-1 z-[100] bg-popover text-popover-foreground rounded-md border shadow-lg p-2 ${className}`}>
    {children}
  </div>
);

// Filter condition types
export type FilterOperator = 'AND' | 'OR';
export type ComparisonOperator = '=' | '!=' | '>' | '<' | '>=' | '<=' | 'contains' | 'in';

export interface FilterCondition {
  id: string;
  field: string;
  operator: ComparisonOperator;
  values: string[];
  numericValue?: number;
}

export interface FilterGroup {
  id: string;
  operator: FilterOperator;
  conditions: FilterCondition[];
}

export interface FilterConfig {
  groups: FilterGroup[];
  globalOperator: FilterOperator;
}

// Field configuration for available filter options
export interface FilterFieldConfig {
  id: string;
  label: string;
  type: 'select' | 'multi-select' | 'number' | 'range';
  options?: { value: string; label: string; count?: number }[];
  min?: number;
  max?: number;
  step?: number;
}

interface EnhancedFacetedFilterProps {
  fields: FilterFieldConfig[];
  initialFilters?: FilterConfig;
  onChange: (filters: FilterConfig) => void;
  onApply?: () => void;
  className?: string;
  compact?: boolean;
}

// Generate unique IDs
const generateId = () => Math.random().toString(36).substring(2, 9);

// Default filter config
const createDefaultFilter = (): FilterConfig => ({
  groups: [{
    id: generateId(),
    operator: 'AND',
    conditions: [],
  }],
  globalOperator: 'AND',
});

// Main component
export function EnhancedFacetedFilter({
  fields,
  initialFilters,
  onChange,
  onApply,
  className,
  compact = false,
}: EnhancedFacetedFilterProps) {
  const [filters, setFilters] = React.useState<FilterConfig>(
    initialFilters || createDefaultFilter()
  );
  const [isExpanded, setIsExpanded] = React.useState(!compact);

  // Update parent when filters change
  React.useEffect(() => {
    onChange(filters);
  }, [filters, onChange]);

  // Add a new condition to a group
  const addCondition = (groupId: string) => {
    setFilters(prev => ({
      ...prev,
      groups: prev.groups.map(group =>
        group.id === groupId
          ? {
              ...group,
              conditions: [
                ...group.conditions,
                {
                  id: generateId(),
                  field: fields[0]?.id || '',
                  operator: 'in',
                  values: [],
                },
              ],
            }
          : group
      ),
    }));
  };

  // Remove a condition
  const removeCondition = (groupId: string, conditionId: string) => {
    setFilters(prev => ({
      ...prev,
      groups: prev.groups.map(group =>
        group.id === groupId
          ? {
              ...group,
              conditions: group.conditions.filter(c => c.id !== conditionId),
            }
          : group
      ),
    }));
  };

  // Update a condition
  const updateCondition = (
    groupId: string,
    conditionId: string,
    updates: Partial<FilterCondition>
  ) => {
    setFilters(prev => ({
      ...prev,
      groups: prev.groups.map(group =>
        group.id === groupId
          ? {
              ...group,
              conditions: group.conditions.map(c =>
                c.id === conditionId ? { ...c, ...updates } : c
              ),
            }
          : group
      ),
    }));
  };

  // Toggle group operator
  const toggleGroupOperator = (groupId: string) => {
    setFilters(prev => ({
      ...prev,
      groups: prev.groups.map(group =>
        group.id === groupId
          ? { ...group, operator: group.operator === 'AND' ? 'OR' : 'AND' }
          : group
      ),
    }));
  };

  // Add a new group
  const addGroup = () => {
    setFilters(prev => ({
      ...prev,
      groups: [
        ...prev.groups,
        {
          id: generateId(),
          operator: 'AND',
          conditions: [],
        },
      ],
    }));
  };

  // Remove a group
  const removeGroup = (groupId: string) => {
    if (filters.groups.length <= 1) return;
    setFilters(prev => ({
      ...prev,
      groups: prev.groups.filter(g => g.id !== groupId),
    }));
  };

  // Toggle global operator
  const toggleGlobalOperator = () => {
    setFilters(prev => ({
      ...prev,
      globalOperator: prev.globalOperator === 'AND' ? 'OR' : 'AND',
    }));
  };

  // Clear all filters
  const clearFilters = () => {
    setFilters(createDefaultFilter());
  };

  // Count active filters
  const activeFilterCount = filters.groups.reduce(
    (count, group) => count + group.conditions.filter(c => c.values.length > 0 || c.numericValue !== undefined).length,
    0
  );

  // Render a single condition
  const renderCondition = (groupId: string, condition: FilterCondition, index: number) => {
    const field = fields.find(f => f.id === condition.field);
    
    return (
      <div key={condition.id} className="flex items-center gap-2 py-2">
        {/* Field selector */}
        <Select
          value={condition.field}
          onValueChange={(value) => updateCondition(groupId, condition.id, { field: value, values: [] })}
        >
          <SelectTrigger className="w-[140px] h-8 text-xs">
            <SelectValue placeholder="Select field" />
          </SelectTrigger>
          <SelectContent>
            {fields.map(f => (
              <SelectItem key={f.id} value={f.id} className="text-xs">
                {f.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* Value selector based on field type */}
        {field?.type === 'multi-select' && (
          <MultiSelectDropdown
            options={field.options || []}
            selected={condition.values}
            onChange={(values) => updateCondition(groupId, condition.id, { values })}
            placeholder="Select values..."
          />
        )}

        {field?.type === 'select' && (
          <Select
            value={condition.values[0] || ''}
            onValueChange={(value) => updateCondition(groupId, condition.id, { values: [value] })}
          >
            <SelectTrigger className="w-[160px] h-8 text-xs">
              <SelectValue placeholder="Select value" />
            </SelectTrigger>
            <SelectContent>
              {field.options?.map(opt => (
                <SelectItem key={opt.value} value={opt.value} className="text-xs">
                  {opt.label} {opt.count !== undefined && `(${opt.count})`}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        {field?.type === 'number' && (
          <div className="flex items-center gap-2">
            <Select
              value={condition.operator}
              onValueChange={(op) => updateCondition(groupId, condition.id, { operator: op as ComparisonOperator })}
            >
              <SelectTrigger className="w-[60px] h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value=">" className="text-xs">&gt;</SelectItem>
                <SelectItem value=">=" className="text-xs">≥</SelectItem>
                <SelectItem value="=" className="text-xs">=</SelectItem>
                <SelectItem value="<=" className="text-xs">≤</SelectItem>
                <SelectItem value="<" className="text-xs">&lt;</SelectItem>
              </SelectContent>
            </Select>
            <input
              type="number"
              className="w-[80px] h-8 px-2 text-xs border rounded-md"
              value={condition.numericValue || ''}
              min={field.min}
              max={field.max}
              step={field.step}
              onChange={(e) => updateCondition(groupId, condition.id, { numericValue: parseFloat(e.target.value) || 0 })}
            />
          </div>
        )}

        {field?.type === 'range' && (
          <div className="flex items-center gap-2 w-[180px]">
            <span className="text-xs text-muted-foreground">{field.min || 0}</span>
            <Slider
              value={[condition.numericValue || field.min || 0]}
              min={field.min || 0}
              max={field.max || 1}
              step={field.step || 0.1}
              onValueChange={([value]) => updateCondition(groupId, condition.id, { numericValue: value })}
              className="flex-1"
            />
            <span className="text-xs font-mono">{(condition.numericValue || 0).toFixed(1)}</span>
          </div>
        )}

        {/* Remove button */}
        <Button
          size="icon"
          variant="ghost"
          className="h-6 w-6 text-muted-foreground hover:text-destructive"
          onClick={() => removeCondition(groupId, condition.id)}
        >
          <X className="h-3 w-3" />
        </Button>
      </div>
    );
  };

  // Compact view - just shows active filter badges
  if (compact && !isExpanded) {
    return (
      <div className={`flex items-center gap-2 flex-wrap ${className}`}>
        <Button
          variant="outline"
          size="sm"
          className="h-8"
          onClick={() => setIsExpanded(true)}
        >
          <Filter className="h-3 w-3 mr-2" />
          Filters
          {activeFilterCount > 0 && (
            <Badge variant="secondary" className="ml-2 h-5 px-1.5">
              {activeFilterCount}
            </Badge>
          )}
        </Button>
        
        {/* Show active filter badges */}
        {filters.groups.map(group =>
          group.conditions
            .filter(c => c.values.length > 0 || c.numericValue !== undefined)
            .map(condition => {
              const field = fields.find(f => f.id === condition.field);
              return (
                <Badge key={condition.id} variant="secondary" className="text-xs">
                  {field?.label}:{' '}
                  {condition.values.length > 0
                    ? condition.values.slice(0, 2).join(', ') + (condition.values.length > 2 ? '...' : '')
                    : `${condition.operator} ${condition.numericValue}`}
                  <button
                    className="ml-1 hover:text-destructive"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeCondition(group.id, condition.id);
                    }}
                  >
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              );
            })
        )}
        
        {activeFilterCount > 0 && (
          <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={clearFilters}>
            Clear all
          </Button>
        )}
      </div>
    );
  }

  return (
    <Card className={className}>
      <CardHeader className="py-3 px-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            <Filter className="h-4 w-4" />
            Advanced Filters
            {activeFilterCount > 0 && (
              <Badge variant="secondary">{activeFilterCount} active</Badge>
            )}
          </CardTitle>
          <div className="flex items-center gap-2">
            {activeFilterCount > 0 && (
              <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={clearFilters}>
                Clear all
              </Button>
            )}
            {compact && (
              <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setIsExpanded(false)}>
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="pt-0 px-4 pb-4">
        <div className="space-y-4">
          {filters.groups.map((group, groupIndex) => (
            <React.Fragment key={group.id}>
              {/* Group separator with operator */}
              {groupIndex > 0 && (
                <div className="flex items-center gap-2 py-2">
                  <div className="flex-1 border-t" />
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-6 px-2 text-xs font-mono"
                    onClick={toggleGlobalOperator}
                  >
                    {filters.globalOperator}
                  </Button>
                  <div className="flex-1 border-t" />
                </div>
              )}
              
              {/* Group container */}
              <div className="border rounded-lg p-3 bg-muted/30">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">
                      Match{' '}
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-5 px-1 text-xs font-mono text-primary"
                        onClick={() => toggleGroupOperator(group.id)}
                      >
                        {group.operator === 'AND' ? 'ALL' : 'ANY'}
                      </Button>
                      {' '}of:
                    </span>
                  </div>
                  
                  {filters.groups.length > 1 && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 text-muted-foreground hover:text-destructive"
                      onClick={() => removeGroup(group.id)}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  )}
                </div>
                
                {/* Conditions */}
                <div className="space-y-1">
                  {group.conditions.map((condition, index) => (
                    <React.Fragment key={condition.id}>
                      {index > 0 && (
                        <div className="flex items-center gap-2 py-1">
                          <span className="text-xs text-muted-foreground pl-2 font-mono">
                            {group.operator}
                          </span>
                        </div>
                      )}
                      {renderCondition(group.id, condition, index)}
                    </React.Fragment>
                  ))}
                </div>
                
                {/* Add condition button */}
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs mt-2"
                  onClick={() => addCondition(group.id)}
                >
                  <Plus className="h-3 w-3 mr-1" />
                  Add condition
                </Button>
              </div>
            </React.Fragment>
          ))}
          
          {/* Add group button */}
          <Button
            variant="outline"
            size="sm"
            className="w-full h-8 text-xs"
            onClick={addGroup}
          >
            <Plus className="h-3 w-3 mr-1" />
            Add filter group
          </Button>
          
          {/* Apply button */}
          {onApply && (
            <Button className="w-full" onClick={onApply}>
              Apply Filters
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// Multi-select dropdown component - simple version that closes on outside click
function MultiSelectDropdown({
  options,
  selected,
  onChange,
  placeholder,
}: {
  options: { value: string; label: string; count?: number }[];
  selected: string[];
  onChange: (values: string[]) => void;
  placeholder?: string;
}) {
  const [isOpen, setIsOpen] = React.useState(false);
  const dropdownRef = React.useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  const toggleValue = (value: string) => {
    if (selected.includes(value)) {
      onChange(selected.filter(v => v !== value));
    } else {
      onChange([...selected, value]);
    }
  };

  return (
    <div ref={dropdownRef} className="relative">
      <Button
        variant="outline"
        role="combobox"
        className="w-[200px] h-8 justify-between text-xs"
        onClick={() => setIsOpen(!isOpen)}
      >
        {selected.length > 0 ? (
          <span className="truncate">
            {selected.length === 1
              ? options.find(o => o.value === selected[0])?.label
              : `${selected.length} selected`}
          </span>
        ) : (
          <span className="text-muted-foreground">{placeholder}</span>
        )}
        <ChevronDown className="ml-2 h-3 w-3 shrink-0 opacity-50" />
      </Button>
      
      {isOpen && (
        <div className="absolute top-full left-0 mt-1 z-[100] w-[200px] bg-popover text-popover-foreground rounded-md border shadow-lg p-2">
          <div className="space-y-1 max-h-[200px] overflow-y-auto">
            {options.map(option => (
              <div
                key={option.value}
                className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-muted cursor-pointer"
                onClick={() => toggleValue(option.value)}
              >
                <input
                  type="checkbox"
                  checked={selected.includes(option.value)}
                  onChange={() => toggleValue(option.value)}
                  className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                />
                <span className="text-xs flex-1">{option.label}</span>
                {option.count !== undefined && (
                  <span className="text-xs text-muted-foreground">{option.count}</span>
                )}
              </div>
            ))}
          </div>
          
          <div className="border-t mt-2 pt-2 flex gap-2">
            {selected.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                className="flex-1 h-7 text-xs"
                onClick={() => onChange([])}
              >
                Clear
              </Button>
            )}
            <Button
              variant="default"
              size="sm"
              className="flex-1 h-7 text-xs"
              onClick={() => setIsOpen(false)}
            >
              Done
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

// Helper to convert FilterConfig to query parameters
export function filtersToQuery(filters: FilterConfig): Record<string, string> {
  const params: Record<string, string[]> = {};
  
  filters.groups.forEach(group => {
    group.conditions.forEach(condition => {
      if (condition.values.length > 0) {
        const key = condition.field;
        params[key] = params[key] || [];
        params[key].push(...condition.values);
      }
      if (condition.numericValue !== undefined) {
        const key = `${condition.field}_${condition.operator}`;
        params[key] = [condition.numericValue.toString()];
      }
    });
  });
  
  // Convert arrays to comma-separated strings
  const result: Record<string, string> = {};
  Object.entries(params).forEach(([key, values]) => {
    result[key] = [...new Set(values)].join(',');
  });
  
  return result;
}

// Pre-built filter configurations for common use cases
export const EXPERIMENT_FILTER_FIELDS: FilterFieldConfig[] = [
  {
    id: 'outcome',
    label: 'Outcome',
    type: 'multi-select',
    options: [
      { value: 'positive', label: 'Positive' },
      { value: 'negative', label: 'Negative' },
      { value: 'inconclusive', label: 'Inconclusive' },
      { value: 'dose_dependent', label: 'Dose-dependent' },
    ],
  },
  {
    id: 'cell_line',
    label: 'Cell Line',
    type: 'multi-select',
    options: [
      { value: 'HeLa', label: 'HeLa' },
      { value: 'A549', label: 'A549' },
      { value: 'MCF7', label: 'MCF7' },
      { value: 'HEK293', label: 'HEK293' },
      { value: 'PC3', label: 'PC3' },
      { value: 'U2OS', label: 'U2OS' },
      { value: 'HCT116', label: 'HCT116' },
      { value: 'A431', label: 'A431' },
      { value: 'H1975', label: 'H1975' },
    ],
  },
  {
    id: 'experiment_type',
    label: 'Experiment Type',
    type: 'multi-select',
    options: [
      { value: 'binding_assay', label: 'Binding Assay' },
      { value: 'activity_assay', label: 'Activity Assay' },
      { value: 'admet', label: 'ADMET' },
      { value: 'phenotypic', label: 'Phenotypic' },
      { value: 'western_blot', label: 'Western Blot' },
      { value: 'microscopy', label: 'Microscopy' },
    ],
  },
  {
    id: 'quality_score',
    label: 'Quality',
    type: 'range',
    min: 0,
    max: 1,
    step: 0.1,
  },
];

export const COMPOUND_FILTER_FIELDS: FilterFieldConfig[] = [
  {
    id: 'modality',
    label: 'Modality',
    type: 'multi-select',
    options: [
      { value: 'drug', label: 'Drug/Small Molecule' },
      { value: 'target', label: 'Target/Protein' },
    ],
  },
  {
    id: 'activity_class',
    label: 'Activity',
    type: 'multi-select',
    options: [
      { value: 'active', label: 'Active' },
      { value: 'inactive', label: 'Inactive' },
      { value: 'intermediate', label: 'Intermediate' },
    ],
  },
  {
    id: 'similarity_score',
    label: 'Min Similarity',
    type: 'range',
    min: 0,
    max: 1,
    step: 0.05,
  },
];
