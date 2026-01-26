"use client"

import { PageHeader, SectionHeader } from "@/components/page-header"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Folder, FileText, Database, HardDrive, Upload, CloudUpload, Eye, Download, Trash2 } from "lucide-react"

export default function DataPage() {
  const datasets = [
    { name: "DrugBank Compounds", type: "Molecules", count: "12,450", size: "45.2 MB", updated: "2024-01-15" },
    { name: "ChEMBL Kinase Inhibitors", type: "Molecules", count: "8,234", size: "32.1 MB", updated: "2024-01-10" },
    { name: "Custom Protein Targets", type: "Proteins", count: "1,245", size: "78.5 MB", updated: "2024-01-08" },
  ]

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <PageHeader
        title="Data Management"
        subtitle="Upload, manage, and organize your datasets"
        icon={<Database className="h-8 w-8" />} 
      />

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[
            { label: "Datasets", value: "5", icon: Folder, color: "text-blue-500" },
            { label: "Molecules", value: "24.5K", icon: FileText, color: "text-cyan-500" },
            { label: "Proteins", value: "1.2K", icon: Database, color: "text-emerald-500" },
            { label: "Storage Used", value: "156 MB", icon: HardDrive, color: "text-amber-500" },
        ].map((stat, i) => (
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
