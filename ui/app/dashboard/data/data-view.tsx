"use client"

import { CloudUpload, Database, Download, Eye, FileText, Folder, HardDrive, Trash2,Upload } from "lucide-react"

import { SectionHeader } from "@/components/page-header"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent,TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Dataset, Statistics } from "@/types/data"

interface DataViewProps {
    datasets: Dataset[];
    stats: Statistics | null;
}

export function DataView({ datasets, stats }: DataViewProps) {
    const statCards = [
        { label: "Datasets", value: stats?.datasets?.toString() ?? "—", icon: Folder, color: "text-blue-500" },
        { label: "Molecules", value: stats?.molecules ?? "—", icon: FileText, color: "text-cyan-500" },
        { label: "Proteins", value: stats?.proteins ?? "—", icon: Database, color: "text-emerald-500" },
        { label: "Storage Used", value: stats?.storage ?? "—", icon: HardDrive, color: "text-amber-500" },
    ]

    return (
        <div className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {statCards.map((stat, i) => (
                    <Card key={i}>
                        <CardContent className="p-6">
                            <div className="flex justify-between items-start mb-2">
                                <div className="text-sm font-medium text-muted-foreground">{stat.label}</div>
                                <div className={`text-lg ${stat.color}`}><stat.icon className="h-5 w-5" /></div>
                            </div>
                            <div className="text-2xl font-bold">{stat.value}</div>
                        </CardContent>
                    </Card>
                ))}
            </div>

            <Tabs defaultValue="datasets">
                <TabsList>
                    <TabsTrigger value="datasets">Datasets</TabsTrigger>
                    <TabsTrigger value="upload">Upload</TabsTrigger>
                    <TabsTrigger value="processing">Processing</TabsTrigger>
                </TabsList>
                <TabsContent value="datasets" className="space-y-4">
                    <SectionHeader title="Your Datasets" icon={<Folder className="h-5 w-5 text-primary" />} />
                    <Card>
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Name</TableHead>
                                    <TableHead>Type</TableHead>
                                    <TableHead>Items</TableHead>
                                    <TableHead>Size</TableHead>
                                    <TableHead>Updated</TableHead>
                                    <TableHead className="text-right">Actions</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {datasets.length === 0 && (
                                     <TableRow>
                                        <TableCell colSpan={6} className="text-center text-muted-foreground text-sm">No datasets found.</TableCell>
                                    </TableRow>
                                )}
                                {datasets.map((ds, i) => (
                                    <TableRow key={i}>
                                        <TableCell className="font-medium">{ds.name}</TableCell>
                                        <TableCell>
                                            <div className="flex items-center gap-2">
                                                <Badge variant={ds.type === 'Molecules' ? 'default' : 'secondary'}>{ds.type}</Badge>
                                            </div>
                                        </TableCell>
                                        <TableCell>{ds.count}</TableCell>
                                        <TableCell>{ds.size}</TableCell>
                                        <TableCell>{ds.updated}</TableCell>
                                        <TableCell className="text-right">
                                            <div className="flex justify-end gap-2">
                                                <Button size="icon" variant="ghost" className="h-8 w-8"><Eye className="h-4 w-4" /></Button>
                                                <Button size="icon" variant="ghost" className="h-8 w-8"><Download className="h-4 w-4" /></Button>
                                                <Button size="icon" variant="ghost" className="h-8 w-8 text-destructive hover:text-destructive"><Trash2 className="h-4 w-4" /></Button>
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </Card>
                </TabsContent>
                <TabsContent value="upload">
                    <SectionHeader title="Upload New Data" icon={<Upload className="h-5 w-5 text-primary" />} />
                    <Card>
                        <CardContent className="p-12">
                            <div className="border-2 border-dashed rounded-lg p-12 flex flex-col items-center justify-center text-center space-y-4 hover:bg-accent/50 transition-colors cursor-pointer">
                                <div className="h-16 w-16 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                                    <CloudUpload className="h-8 w-8" />
                                </div>
                                <div>
                                    <div className="text-lg font-semibold">Click to upload or drag and drop</div>
                                    <div className="text-sm text-muted-foreground">CSV, SDF, FASTA, or JSON (max 50MB)</div>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>
                <TabsContent value="processing">
                    <Card>
                        <CardContent className="p-12 text-center text-muted-foreground">
                            No active processing tasks.
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    )
}
