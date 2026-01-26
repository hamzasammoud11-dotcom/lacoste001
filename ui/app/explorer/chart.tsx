"use client"

import { AlertCircle } from "lucide-react"
import { Cell,ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts"

import { Card, CardContent } from "@/components/ui/card"
import { DataPoint } from "@/types/explorer"

export function ExplorerChart({ data }: { data: DataPoint[] }) {
    if (!data || data.length === 0) {
        return (
            <div className="flex h-full items-center justify-center text-muted-foreground">
                <AlertCircle className="mr-2 h-4 w-4" />
                No data available for this configuration.
            </div>
        )
    }

    return (
        <Card className="h-[500px] overflow-hidden bg-gradient-to-br from-card to-secondary/30">
            <CardContent className="p-4 h-full relative">
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <XAxis type="number" dataKey="x" name="PC1" stroke="currentColor" fontSize={12} tickLine={false} axisLine={{ strokeOpacity: 0.2 }} />
                        <YAxis type="number" dataKey="y" name="PC2" stroke="currentColor" fontSize={12} tickLine={false} axisLine={{ strokeOpacity: 0.2 }} />
                        <ZAxis type="number" dataKey="z" range={[50, 400]} />
                        <Tooltip 
                            cursor={{ strokeDasharray: '3 3' }} 
                            content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                    const point = payload[0].payload as DataPoint;
                                    return (
                                        <div className="bg-popover border border-border p-3 rounded-lg shadow-xl text-sm z-50">
                                            <p className="font-bold mb-1 text-primary">{point.name}</p>
                                            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted-foreground">
                                                <span>X:</span> <span className="text-foreground">{Number(point.x).toFixed(2)}</span>
                                                <span>Y:</span> <span className="text-foreground">{Number(point.y).toFixed(2)}</span>
                                                <span>Affinity:</span> <span className="text-foreground">{Number(point.affinity).toFixed(2)}</span>
                                            </div>
                                        </div>
                                    )
                                }
                                return null;
                            }} 
                        />
                        <Scatter name="Molecules" data={data} fill="#8884d8" animationDuration={1000}>
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} fillOpacity={0.7} className="hover:opacity-100 transition-opacity duration-200" />
                            ))}
                        </Scatter>
                    </ScatterChart>
                </ResponsiveContainer>
            </CardContent>
        </Card>
    )
}
