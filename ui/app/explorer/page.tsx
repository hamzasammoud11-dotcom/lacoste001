"use client"

import * as React from "react"
import { PageHeader, SectionHeader } from "@/components/page-header"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { Dna, RefreshCw, Map as MapIcon, BarChart2, Activity, Zap, Grid3X3 } from "lucide-react"
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, Cell } from "recharts"

interface DataPoint {
    x: number;
    y: number;
    z: number;
    color: string;
    name: string;
    affinity: number;
}

export default function ExplorerPage() {
  const [dataset, setDataset] = React.useState("DrugBank")
  const [visualization, setVisualization] = React.useState("UMAP")
  const [colorBy, setColorBy] = React.useState("Activity")
  const [data, setData] = React.useState<DataPoint[]>([])
  
  // Sample Data Generation
  React.useEffect(() => {
    const points: DataPoint[] = []
    const clusters = [
        { cx: 2, cy: 3, color: "var(--color-chart-1)" },
        { cx: -2, cy: -1, color: "var(--color-chart-2)" },
        { cx: 4, cy: -2, color: "var(--color-chart-3)" },
        { cx: -1, cy: 4, color: "var(--color-chart-4)" }
    ]
    
    for (let i = 0; i < 200; i++) {
        const cluster = clusters[Math.floor(i / 50)]
        points.push({
            x: cluster.cx + (Math.random() - 0.5) * 2,
            y: cluster.cy + (Math.random() - 0.5) * 2,
            z: Math.random() * 100,
            color: cluster.color,
            name: `Mol_${i}`,
            affinity: Math.random() * 100
        })
    }
    setData(points)
  }, [dataset]) 

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
        <PageHeader 
            title="Data Explorer" 
            subtitle="Visualize molecular embeddings and relationships" 
            icon={<Dna className="h-8 w-8 text-primary" />}
        />

        <Card className="border-l-4 border-l-primary/50">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5" />
                    Visualization Controls
                </CardTitle>
                <CardDescription>Adjust projection parameters and visual styles</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="space-y-2">
                    <Label htmlFor="dataset">Dataset</Label>
                    <Select value={dataset} onValueChange={setDataset}>
                        <SelectTrigger id="dataset">
                            <SelectValue placeholder="Select dataset" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="DrugBank">DrugBank</SelectItem>
                            <SelectItem value="ChEMBL">ChEMBL</SelectItem>
                            <SelectItem value="ZINC">ZINC</SelectItem>
                        </SelectContent>
                    </Select>
                </div>
                <div className="space-y-2">
                    <Label htmlFor="visualization">Algorithm</Label>
                    <Select value={visualization} onValueChange={setVisualization}>
                        <SelectTrigger id="visualization">
                            <SelectValue placeholder="Select algorithm" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="UMAP">UMAP</SelectItem>
                            <SelectItem value="t-SNE">t-SNE</SelectItem>
                            <SelectItem value="PCA">PCA</SelectItem>
                        </SelectContent>
                    </Select>
                </div>
                <div className="space-y-2">
                    <Label htmlFor="colorBy">Color Mapping</Label>
                    <Select value={colorBy} onValueChange={setColorBy}>
                        <SelectTrigger id="colorBy">
                            <SelectValue placeholder="Select color metric" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="Activity">Binding Affinity</SelectItem>
                            <SelectItem value="MW">Molecular Weight</SelectItem>
                            <SelectItem value="LogP">LogP</SelectItem>
                            <SelectItem value="Cluster">Cluster ID</SelectItem>
                        </SelectContent>
                    </Select>
                </div>
                <div className="flex items-end">
                    <Button variant="secondary" className="w-full" onClick={() => setDataset(d => d === "DrugBank" ? "ChEMBL" : "DrugBank")}>
                        <RefreshCw className="mr-2 h-4 w-4" />
                        Regenerate View
                    </Button>
                </div>
            </CardContent>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-4">
                <SectionHeader title="Embedding Space" icon={<MapIcon className="h-5 w-5 text-primary" />} />
                <Card className="h-[500px] overflow-hidden bg-gradient-to-br from-card to-secondary/30">
                    <CardContent className="p-4 h-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <XAxis type="number" dataKey="x" name="PC1" stroke="currentColor" fontSize={12} tickLine={false} axisLine={{ strokeOpacity: 0.2 }} />
                                <YAxis type="number" dataKey="y" name="PC2" stroke="currentColor" fontSize={12} tickLine={false} axisLine={{ strokeOpacity: 0.2 }} />
                                <ZAxis type="number" dataKey="z" range={[50, 400]} />
                                <Tooltip 
                                    cursor={{ strokeDasharray: '3 3' }} 
                                    content={({ active, payload }) => {
                                        if (active && payload && payload.length) {
                                            const data = payload[0].payload;
                                            return (
                                                <div className="bg-popover border border-border p-3 rounded-lg shadow-xl text-sm">
                                                    <p className="font-bold mb-1 text-primary">{data.name}</p>
                                                    <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted-foreground">
                                                        <span>X:</span> <span className="text-foreground">{Number(data.x).toFixed(2)}</span>
                                                        <span>Y:</span> <span className="text-foreground">{Number(data.y).toFixed(2)}</span>
                                                        <span>Affinity:</span> <span className="text-foreground">{Number(data.affinity).toFixed(2)}</span>
                                                    </div>
                                                </div>
                                            )
                                        }
                                        return null;
                                    }} 
                                />
                                <Scatter name="Molecules" data={data} fill="#8884d8">
                                    {data.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} fillOpacity={0.7} className="hover:opacity-100 transition-opacity duration-200" />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
            </div>

            <div className="space-y-6">
                 <SectionHeader title="Data Intelligence" icon={<Grid3X3 className="h-5 w-5 text-primary" />} />
                 
                 <div className="grid grid-cols-1 gap-4">
                    <Card>
                        <CardHeader className="pb-2">
                             <CardTitle className="text-sm font-medium text-muted-foreground flex items-center justify-between">
                                 Active Molecules
                                 <Zap className="h-4 w-4 text-yellow-500" />
                             </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">12,450</div>
                            <p className="text-xs text-muted-foreground mt-1">+2.5% from last month</p>
                        </CardContent>
                    </Card>
                    <Card>
                        <CardHeader className="pb-2">
                             <CardTitle className="text-sm font-medium text-muted-foreground flex items-center justify-between">
                                 Clusters Identified
                                 <Grid3X3 className="h-4 w-4 text-blue-500" />
                             </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-chart-2">4</div>
                            <p className="text-xs text-muted-foreground mt-1">Distinct chemical series</p>
                        </CardContent>
                    </Card>
                    <Card>
                        <CardHeader className="pb-2">
                             <CardTitle className="text-sm font-medium text-muted-foreground flex items-center justify-between">
                                 Avg Confidence
                                 <BarChart2 className="h-4 w-4 text-green-500" />
                             </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-chart-3">0.89</div>
                            <p className="text-xs text-muted-foreground mt-1">Across all predictions</p>
                        </CardContent>
                    </Card>
                 </div>
                 
                 <div className="p-4 rounded-lg bg-secondary/50 border border-secondary">
                     <h4 className="font-semibold mb-2 text-sm flex items-center gap-2">
                         <Zap className="h-3 w-3 text-primary" />
                         Quick Actions
                     </h4>
                     <div className="space-y-2">
                         <Button variant="outline" size="sm" className="w-full justify-start text-xs">Exort Selection as CSV</Button>
                         <Button variant="outline" size="sm" className="w-full justify-start text-xs">Run Clustering Analysis</Button>
                     </div>
                 </div>
            </div>
        </div>
    </div>
  )
}
