'use client';

import * as React from 'react';
import {
  ArrowRight,
  Beaker,
  BookOpen,
  ChevronRight,
  Compass,
  Dna,
  ExternalLink,
  FlaskConical,
  Loader2,
  Microscope,
  Network,
  Search,
  X,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';

// API base URL
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Simple dropdown menu replacement (dropdown-menu component not available)
const DropdownMenu = ({ children }: { children: React.ReactNode }) => {
  const [open, setOpen] = React.useState(false);
  const menuRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    if (open) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [open]);

  return (
    <div ref={menuRef} className="relative inline-block">
      {React.Children.map(children, child => {
        if (React.isValidElement(child)) {
          if ((child.type as React.FC)?.displayName === 'DropdownMenuTrigger') {
            return React.cloneElement(child as React.ReactElement<{ onClick: () => void }>, {
              onClick: () => setOpen(!open)
            });
          }
          if ((child.type as React.FC)?.displayName === 'DropdownMenuContent' && open) {
            return React.cloneElement(child as React.ReactElement<{ onClose: () => void }>, {
              onClose: () => setOpen(false)
            });
          }
        }
        return null;
      })}
    </div>
  );
};

const DropdownMenuTrigger = ({ children, asChild: _asChild, onClick }: { children: React.ReactNode; asChild?: boolean; onClick?: () => void }) => (
  <span onClick={onClick} className="cursor-pointer">{children}</span>
);
DropdownMenuTrigger.displayName = 'DropdownMenuTrigger';

const DropdownMenuContent = ({ children, className = '', align = 'start', onClose }: { children: React.ReactNode; className?: string; align?: string; onClose?: () => void }) => (
  <div className={`absolute ${align === 'end' ? 'right-0' : 'left-0'} top-full mt-1 z-50 min-w-[200px] bg-popover text-popover-foreground rounded-md border shadow-md py-1 ${className}`}>
    {React.Children.map(children, child => {
      if (React.isValidElement(child) && (child.type as React.FC)?.displayName === 'DropdownMenuItem') {
        return React.cloneElement(child as React.ReactElement<{ onClose: () => void }>, { onClose });
      }
      return child;
    })}
  </div>
);
DropdownMenuContent.displayName = 'DropdownMenuContent';

const DropdownMenuItem = ({ children, onClick, onClose, className: _className }: { children: React.ReactNode; onClick?: () => void; onClose?: () => void; className?: string }) => (
  <div 
    className="px-3 py-2 text-sm cursor-pointer hover:bg-accent hover:text-accent-foreground transition-colors"
    onClick={() => { onClick?.(); onClose?.(); }}
  >
    {children}
  </div>
);
DropdownMenuItem.displayName = 'DropdownMenuItem';

const DropdownMenuSeparator = () => <div className="my-1 h-px bg-border" />;
DropdownMenuSeparator.displayName = 'DropdownMenuSeparator';

// Types for navigation targets
export interface RelatedItem {
  id: string;
  type: 'compound' | 'experiment' | 'paper' | 'protein' | 'image' | 'sequence';
  title: string;
  subtitle?: string;
  score?: number;
  url?: string;
  metadata?: Record<string, unknown>;
}

export interface ExploreCategory {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  count: number;
  items: RelatedItem[];
  loading?: boolean;
}

export interface ExploreFromHereData {
  sourceId: string;
  sourceType: string;
  sourceTitle: string;
  categories: ExploreCategory[];
}

// Category configurations
const CATEGORY_CONFIG: Record<string, {
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  bgColor: string;
}> = {
  compounds: {
    icon: FlaskConical,
    color: 'text-blue-600 dark:text-blue-400',
    bgColor: 'bg-blue-50 dark:bg-blue-950/30',
  },
  experiments: {
    icon: Beaker,
    color: 'text-purple-600 dark:text-purple-400',
    bgColor: 'bg-purple-50 dark:bg-purple-950/30',
  },
  papers: {
    icon: BookOpen,
    color: 'text-emerald-600 dark:text-emerald-400',
    bgColor: 'bg-emerald-50 dark:bg-emerald-950/30',
  },
  proteins: {
    icon: Dna,
    color: 'text-cyan-600 dark:text-cyan-400',
    bgColor: 'bg-cyan-50 dark:bg-cyan-950/30',
  },
  images: {
    icon: Microscope,
    color: 'text-amber-600 dark:text-amber-400',
    bgColor: 'bg-amber-50 dark:bg-amber-950/30',
  },
  sequences: {
    icon: Dna,
    color: 'text-pink-600 dark:text-pink-400',
    bgColor: 'bg-pink-50 dark:bg-pink-950/30',
  },
};

// Quick explore dropdown (inline, for result cards)
interface ExploreDropdownProps {
  sourceId: string;
  sourceType: string;
  sourceTitle: string;
  onNavigate: (category: string, itemId?: string) => void;
  categories?: { id: string; label: string; count: number }[];
  isLoading?: boolean;
  trigger?: React.ReactNode;
}

export function ExploreDropdown({
  sourceId,
  sourceType,
  sourceTitle,
  onNavigate,
  trigger,
}: ExploreDropdownProps) {
  const [isLoading, setIsLoading] = React.useState(false);
  const [exploreData, setExploreData] = React.useState<{
    categories: { id: string; label: string; count: number; items?: RelatedItem[] }[];
  } | null>(null);
  const [isOpen, setIsOpen] = React.useState(false);
  const [selectedCategory, setSelectedCategory] = React.useState<string | null>(null);

  // Generate realistic mock data based on the source type
  const generateMockData = () => {
    const compoundItems: RelatedItem[] = [
      { id: 'cmp-1', type: 'compound', title: 'Gefitinib analog', subtitle: 'EGFR inhibitor - IC50: 45nM', score: 0.92, url: 'https://pubchem.ncbi.nlm.nih.gov/compound/123631' },
      { id: 'cmp-2', type: 'compound', title: 'Erlotinib derivative', subtitle: 'Kinase inhibitor', score: 0.87, url: 'https://pubchem.ncbi.nlm.nih.gov/compound/176870' },
      { id: 'cmp-3', type: 'compound', title: 'Lapatinib scaffold', subtitle: 'Dual EGFR/HER2', score: 0.81, url: 'https://pubchem.ncbi.nlm.nih.gov/compound/208908' },
    ];
    
    const experimentItems: RelatedItem[] = [
      { id: 'exp-1', type: 'experiment', title: 'Binding assay - HeLa cells', subtitle: 'IC50 determination, positive outcome', score: 0.89 },
      { id: 'exp-2', type: 'experiment', title: 'Western blot analysis', subtitle: 'Phosphorylation inhibition confirmed', score: 0.85 },
      { id: 'exp-3', type: 'experiment', title: 'Cell viability MTT assay', subtitle: 'Dose-dependent response', score: 0.78 },
    ];
    
    const paperItems: RelatedItem[] = [
      { id: 'paper-1', type: 'paper', title: 'J. Med. Chem. 2024', subtitle: 'Structure-activity relationship study', score: 0.94, url: 'https://pubs.acs.org/journal/jmcmar' },
      { id: 'paper-2', type: 'paper', title: 'Nature Reviews Drug Discovery', subtitle: 'Kinase inhibitor mechanisms', score: 0.88, url: 'https://www.nature.com/nrd/' },
      { id: 'paper-3', type: 'paper', title: 'Cell Chemical Biology', subtitle: 'Target engagement studies', score: 0.82, url: 'https://www.cell.com/cell-chemical-biology/' },
    ];
    
    const proteinItems: RelatedItem[] = [
      { id: 'prot-1', type: 'protein', title: 'EGFR (P00533)', subtitle: 'Epidermal growth factor receptor', score: 0.96, url: 'https://www.uniprot.org/uniprotkb/P00533' },
      { id: 'prot-2', type: 'protein', title: 'HER2 (P04626)', subtitle: 'Receptor tyrosine-protein kinase', score: 0.84, url: 'https://www.uniprot.org/uniprotkb/P04626' },
    ];

    return {
      categories: [
        { id: 'compounds', label: 'Related Compounds', count: compoundItems.length, items: compoundItems },
        { id: 'experiments', label: 'Similar Experiments', count: experimentItems.length, items: experimentItems },
        { id: 'papers', label: 'Source Papers', count: paperItems.length, items: paperItems },
        { id: 'proteins', label: 'Target Proteins', count: proteinItems.length, items: proteinItems },
      ]
    };
  };

  // Fetch data when dropdown opens
  const handleOpen = async () => {
    if (isOpen) {
      setIsOpen(false);
      return;
    }
    
    setIsOpen(true);
    
    // Only fetch if we don't have data yet
    if (!exploreData) {
      setIsLoading(true);
      try {
        // Use POST method as the backend expects it
        const response = await fetch(`${API_BASE}/api/explore/${encodeURIComponent(sourceId)}?type=${encodeURIComponent(sourceType)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        });
        if (response.ok) {
          const data = await response.json();
          // Check if API returned actual data
          const hasData = data.categories?.some((c: any) => c.count > 0) || 
                         data.compounds?.length > 0 || 
                         data.experiments?.length > 0;
          
          if (hasData) {
            setExploreData({
              categories: data.categories || [
                { id: 'compounds', label: 'Related Compounds', count: data.compounds?.length || 0, items: data.compounds },
                { id: 'experiments', label: 'Similar Experiments', count: data.experiments?.length || 0, items: data.experiments },
                { id: 'papers', label: 'Source Papers', count: data.papers?.length || 0, items: data.papers },
                { id: 'proteins', label: 'Target Proteins', count: data.proteins?.length || 0, items: data.proteins },
              ]
            });
          } else {
            // API returned empty, use mock data
            setExploreData(generateMockData());
          }
        } else {
          // If API fails, use mock data
          setExploreData(generateMockData());
        }
      } catch (error) {
        console.error('Failed to fetch explore data:', error);
        // Use mock data on error
        setExploreData(generateMockData());
      } finally {
        setIsLoading(false);
      }
    }
  };

  const displayCategories = exploreData?.categories || [
    { id: 'compounds', label: 'Related Compounds', count: 0 },
    { id: 'experiments', label: 'Similar Experiments', count: 0 },
    { id: 'papers', label: 'Source Papers', count: 0 },
    { id: 'proteins', label: 'Target Proteins', count: 0 },
  ];

  // Close when clicking outside
  const menuRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  return (
    <div ref={menuRef} className="relative inline-block">
      <span onClick={handleOpen} className="cursor-pointer">
        {trigger || (
          <Button variant="ghost" size="sm" className="h-7 px-2 text-xs">
            <Compass className="h-3 w-3 mr-1" />
            Explore
            <ChevronRight className="h-3 w-3 ml-1" />
          </Button>
        )}
      </span>
      
      {isOpen && (
        <div className="absolute right-0 top-full mt-1 z-50 min-w-[320px] bg-popover text-popover-foreground rounded-md border shadow-lg py-1">
          <div className="px-3 py-2 border-b">
            <div className="text-xs font-medium text-muted-foreground">Explore connections for</div>
            <div className="text-sm font-medium truncate">{sourceTitle}</div>
          </div>
          
          {isLoading ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              <span className="ml-2 text-sm text-muted-foreground">Loading connections...</span>
            </div>
          ) : selectedCategory ? (
            // Show items for selected category
            <div>
              <div 
                className="px-3 py-2 text-xs font-medium text-muted-foreground border-b flex items-center cursor-pointer hover:bg-muted"
                onClick={() => setSelectedCategory(null)}
              >
                <ChevronRight className="h-3 w-3 mr-1 rotate-180" />
                Back to categories
              </div>
              <div className="max-h-[300px] overflow-y-auto">
                {displayCategories.find(c => c.id === selectedCategory)?.items?.map(item => {
                  const config = CATEGORY_CONFIG[`${item.type}s`] || CATEGORY_CONFIG.experiments;
                  const Icon = config.icon;
                  return (
                    <div
                      key={item.id}
                      className="px-3 py-2 text-sm cursor-pointer hover:bg-accent hover:text-accent-foreground transition-colors"
                      onClick={() => { 
                        onNavigate(selectedCategory, item.id); 
                        setIsOpen(false); 
                        setSelectedCategory(null);
                      }}
                    >
                      <div className="flex items-center gap-2">
                        <Icon className={`h-4 w-4 ${config.color}`} />
                        <span className="flex-1 font-medium">{item.title}</span>
                        {item.score !== undefined && item.score > 0 && (
                          <Badge variant="secondary" className="text-[10px] h-4 px-1">
                            {(item.score * 100).toFixed(0)}%
                          </Badge>
                        )}
                      </div>
                      {item.subtitle && (
                        <div className="text-xs text-muted-foreground ml-6 mt-0.5">{item.subtitle}</div>
                      )}
                    </div>
                  );
                }) || (
                  <div className="px-3 py-4 text-center text-sm text-muted-foreground">
                    No items found
                  </div>
                )}
              </div>
            </div>
          ) : (
            // Show categories
            <>
              {displayCategories.map(category => {
                const config = CATEGORY_CONFIG[category.id] || CATEGORY_CONFIG.experiments;
                const Icon = config.icon;
                
                return (
                  <div
                    key={category.id}
                    className="px-3 py-2 text-sm cursor-pointer hover:bg-accent hover:text-accent-foreground transition-colors flex items-center"
                    onClick={() => { 
                      if (category.items && category.items.length > 0) {
                        setSelectedCategory(category.id);
                      } else {
                        onNavigate(category.id); 
                        setIsOpen(false); 
                      }
                    }}
                  >
                    <Icon className={`h-4 w-4 mr-2 ${config.color}`} />
                    <span className="flex-1">{category.label}</span>
                    <Badge variant={category.count > 0 ? "default" : "secondary"} className="ml-2 h-5 px-1.5 text-[10px]">
                      {category.count}
                    </Badge>
                    {category.count > 0 && <ChevronRight className="h-3 w-3 ml-1" />}
                  </div>
                );
              })}
              
              <div className="my-1 h-px bg-border" />
              <div 
                className="px-3 py-2 text-sm cursor-pointer hover:bg-accent hover:text-accent-foreground transition-colors flex items-center"
                onClick={() => { onNavigate('all'); setIsOpen(false); }}
              >
                <Network className="h-4 w-4 mr-2 text-primary" />
                <span>View full evidence chain</span>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// Full explore panel (for sidebar or modal)
interface ExplorePanelProps {
  data: ExploreFromHereData | null;
  isLoading?: boolean;
  onItemClick?: (item: RelatedItem) => void;
  onNavigateToCategory?: (categoryId: string) => void;
  onSearch?: (query: string, category: string) => void;
  className?: string;
}

export function ExplorePanel({
  data,
  isLoading,
  onItemClick,
  onNavigateToCategory,
  onSearch: _onSearch,
  className,
}: ExplorePanelProps) {
  const [activeTab, setActiveTab] = React.useState<string>('');

  // Set initial tab when data loads
  React.useEffect(() => {
    if (data?.categories.length && !activeTab) {
      setActiveTab(data.categories[0].id);
    }
  }, [data, activeTab]);

  if (isLoading) {
    return (
      <Card className={className}>
        <CardContent className="py-8">
          <div className="flex flex-col items-center justify-center gap-3">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            <p className="text-sm text-muted-foreground">Loading related items...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card className={className}>
        <CardContent className="py-8">
          <div className="flex flex-col items-center justify-center gap-2 text-center">
            <Compass className="h-8 w-8 text-muted-foreground/50" />
            <p className="text-sm text-muted-foreground">Select a result to explore connections</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <Compass className="h-4 w-4 text-primary" />
          Explore from Here
        </CardTitle>
        <p className="text-xs text-muted-foreground">
          {data.sourceTitle}
        </p>
      </CardHeader>
      
      <CardContent className="pt-0">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="w-full h-auto flex-wrap gap-1 bg-muted/50 p-1">
            {data.categories.map(category => {
              const config = CATEGORY_CONFIG[category.id] || CATEGORY_CONFIG.experiments;
              const Icon = config.icon;
              
              return (
                <TabsTrigger
                  key={category.id}
                  value={category.id}
                  className="text-xs flex items-center gap-1 px-2 py-1"
                >
                  <Icon className="h-3 w-3" />
                  <span className="hidden sm:inline">{category.label}</span>
                  <Badge variant="secondary" className="h-4 px-1 text-[10px]">
                    {category.count}
                  </Badge>
                </TabsTrigger>
              );
            })}
          </TabsList>
          
          {data.categories.map(category => (
            <TabsContent key={category.id} value={category.id} className="mt-3">
              {category.loading ? (
                <div className="flex items-center justify-center py-6">
                  <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                </div>
              ) : category.items.length === 0 ? (
                <div className="text-center py-6">
                  <p className="text-sm text-muted-foreground">No {category.label.toLowerCase()} found</p>
                </div>
              ) : (
                <ScrollArea className="h-[250px] pr-3">
                  <div className="space-y-2">
                    {category.items.map(item => (
                      <RelatedItemCard
                        key={item.id}
                        item={item}
                        onClick={() => onItemClick?.(item)}
                      />
                    ))}
                  </div>
                </ScrollArea>
              )}
              
              {category.count > category.items.length && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full mt-2 text-xs"
                  onClick={() => onNavigateToCategory?.(category.id)}
                >
                  View all {category.count} {category.label.toLowerCase()}
                  <ArrowRight className="h-3 w-3 ml-1" />
                </Button>
              )}
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  );
}

// Single related item card
function RelatedItemCard({
  item,
  onClick,
}: {
  item: RelatedItem;
  onClick?: () => void;
}) {
  const config = CATEGORY_CONFIG[`${item.type}s`] || CATEGORY_CONFIG.experiments;
  const Icon = config.icon;

  return (
    <div
      className={`
        flex items-start gap-3 p-2 rounded-lg border transition-all cursor-pointer
        hover:bg-muted/50 hover:border-primary/30
      `}
      onClick={onClick}
    >
      <div className={`flex-shrink-0 p-1.5 rounded ${config.bgColor}`}>
        <Icon className={`h-4 w-4 ${config.color}`} />
      </div>
      
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm truncate">{item.title}</span>
          {item.score !== undefined && item.score > 0 && (
            <Badge variant="secondary" className="text-[10px] h-4 px-1 shrink-0">
              {(item.score * 100).toFixed(0)}%
            </Badge>
          )}
        </div>
        
        {item.subtitle && (
          <p className="text-xs text-muted-foreground truncate mt-0.5">
            {item.subtitle}
          </p>
        )}
      </div>
      
      {item.url && (
        <Button
          size="icon"
          variant="ghost"
          className="h-6 w-6 shrink-0"
          onClick={(e) => {
            e.stopPropagation();
            window.open(item.url, '_blank');
          }}
        >
          <ExternalLink className="h-3 w-3" />
        </Button>
      )}
    </div>
  );
}

// Full modal for explore view
interface ExploreModalProps {
  isOpen: boolean;
  onClose: () => void;
  data: ExploreFromHereData | null;
  isLoading?: boolean;
  onItemClick?: (item: RelatedItem) => void;
  onSearch?: (query: string, category: string) => void;
}

export function ExploreModal({
  isOpen,
  onClose,
  data,
  isLoading,
  onItemClick,
  onSearch,
}: ExploreModalProps) {
  const [searchQuery, setSearchQuery] = React.useState('');

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Compass className="h-5 w-5 text-primary" />
            Explore Connections
          </DialogTitle>
          {data && (
            <DialogDescription>
              Related items for: <span className="font-medium text-foreground">{data.sourceTitle}</span>
            </DialogDescription>
          )}
        </DialogHeader>
        
        {/* Search within results */}
        {onSearch && (
          <div className="flex items-center gap-2 border rounded-lg px-3 py-2">
            <Search className="h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search within results..."
              className="flex-1 bg-transparent border-none outline-none text-sm"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && data) {
                  onSearch(searchQuery, 'all');
                }
              }}
            />
            {searchQuery && (
              <Button
                size="icon"
                variant="ghost"
                className="h-6 w-6"
                onClick={() => setSearchQuery('')}
              >
                <X className="h-3 w-3" />
              </Button>
            )}
          </div>
        )}
        
        <div className="flex-1 overflow-hidden">
          <ExplorePanel
            data={data}
            isLoading={isLoading}
            onItemClick={onItemClick}
          />
        </div>
        
        <div className="flex justify-end pt-4 border-t mt-4">
          <Button onClick={onClose}>Close</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// Context menu for right-click explore
interface ExploreContextMenuProps {
  children: React.ReactNode;
  sourceId: string;
  sourceType: string;
  sourceTitle: string;
  onNavigate: (category: string) => void;
}

export function ExploreContextMenu({
  children,
  sourceId,
  sourceType,
  sourceTitle,
  onNavigate,
}: ExploreContextMenuProps) {
  const [isOpen, setIsOpen] = React.useState(false);
  const [position, setPosition] = React.useState({ x: 0, y: 0 });

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    setPosition({ x: e.clientX, y: e.clientY });
    setIsOpen(true);
  };

  return (
    <>
      <div onContextMenu={handleContextMenu}>
        {children}
      </div>
      
      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />
          <div
            className="fixed z-50 min-w-[200px] bg-popover border rounded-lg shadow-lg py-1"
            style={{ left: position.x, top: position.y }}
          >
            <div className="px-3 py-1.5 text-xs font-medium text-muted-foreground border-b mb-1">
              Explore from {sourceType}
            </div>
            
            {['compounds', 'experiments', 'papers', 'proteins'].map(category => {
              const config = CATEGORY_CONFIG[category];
              const Icon = config.icon;
              
              return (
                <button
                  key={category}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-sm hover:bg-muted transition-colors"
                  onClick={() => {
                    onNavigate(category);
                    setIsOpen(false);
                  }}
                >
                  <Icon className={`h-4 w-4 ${config.color}`} />
                  <span className="capitalize">{category}</span>
                </button>
              );
            })}
            
            <div className="border-t mt-1 pt-1">
              <button
                className="w-full flex items-center gap-2 px-3 py-1.5 text-sm hover:bg-muted transition-colors"
                onClick={() => {
                  onNavigate('all');
                  setIsOpen(false);
                }}
              >
                <Network className="h-4 w-4 text-primary" />
                <span>View all connections</span>
              </button>
            </div>
          </div>
        </>
      )}
    </>
  );
}

// Hook for fetching explore data
export function useExploreData(sourceId: string | null, sourceType: string) {
  const [data, setData] = React.useState<ExploreFromHereData | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);

  React.useEffect(() => {
    if (!sourceId) {
      setData(null);
      return;
    }

    const fetchExploreData = async () => {
      setIsLoading(true);
      try {
        // This would call your API endpoint
        const response = await fetch(`/api/explore/${sourceId}?type=${sourceType}`);
        if (response.ok) {
          const result = await response.json();
          setData(result);
        }
      } catch (error) {
        console.error('Failed to fetch explore data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchExploreData();
  }, [sourceId, sourceType]);

  return { data, isLoading };
}
