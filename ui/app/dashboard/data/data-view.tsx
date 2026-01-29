'use client';

import {
  BarChart2,
  CloudUpload,
  Database,
  Download,
  Eye,
  FileText,
  Folder,
  HardDrive,
  Loader2,
  Trash2,
  Upload,
} from 'lucide-react';
import { useEffect, useState } from 'react';

import { SectionHeader } from '@/components/page-header';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { batchIngest as apiBatchIngest, getIngestJob, startIngestion as apiStartIngestion } from '@/lib/api';
import { Dataset, Statistics } from '@/schemas/data';

interface DataViewProps {
  datasets: Dataset[];
  stats: Statistics | null;
}

interface IngestionJob {
  job_id: string;
  status: string;
  source?: string;
  type?: string;
}

export function DataView({ datasets, stats }: DataViewProps) {
  const [ingestSource, setIngestSource] = useState('pubmed');
  const [ingestQuery, setIngestQuery] = useState('');
  const [ingestLimit, setIngestLimit] = useState(50);
  const [ingestSync, setIngestSync] = useState(false);
  const [ingestCollection, setIngestCollection] = useState('');
  const [ingestError, setIngestError] = useState<string | null>(null);
  const [ingestMessage, setIngestMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [jobs, setJobs] = useState<IngestionJob[]>([]);

  useEffect(() => {
    if (jobs.length === 0) return;
    const interval = setInterval(async () => {
      try {
        const updated = await Promise.all(
          jobs.map(async (job) => {
            try {
              return await getIngestJob(job.job_id);
            } catch {
              return job;
            }
          }),
        );
        setJobs(updated);
      } catch {
        // ignore polling errors
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [jobs]);

  const startIngestion = async () => {
    setIsSubmitting(true);
    setIngestError(null);
    setIngestMessage(null);
    try {
      const payload: Record<string, string | number | boolean> =
        ingestSource === 'all'
          ? {
            query: ingestQuery,
            pubmed_limit: ingestLimit,
            uniprot_limit: ingestLimit,
            chembl_limit: ingestLimit,
            sync: ingestSync,
          }
          : { query: ingestQuery, limit: ingestLimit, sync: ingestSync };

      if (ingestCollection.trim()) {
        payload.collection = ingestCollection.trim();
      }

      const data = await apiStartIngestion(ingestSource, payload);

      if (data.job_id) {
        setJobs((prev) => [
          { job_id: data.job_id, status: 'pending' },
          ...prev,
        ]);
        setIngestMessage(`Ingestion started: ${data.job_id}`);
      } else if (data.result) {
        setIngestMessage('Ingestion completed successfully');
      }
    } catch (err) {
      setIngestError(err instanceof Error ? err.message : 'Ingestion failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleBatchUpload = async (file?: File) => {
    if (!file) return;
    setIsSubmitting(true);
    setIngestError(null);
    setIngestMessage(null);
    try {
      const text = await file.text();
      const items = JSON.parse(text);
      if (!Array.isArray(items)) {
        throw new Error('JSON must be an array of records');
      }
      const data = await apiBatchIngest(items);
      setIngestMessage(`Batch ingested: ${data.ingested || 0} items`);
    } catch (err) {
      setIngestError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsSubmitting(false);
    }
  };
  const statCards = [
    {
      label: 'Datasets',
      value: stats?.datasets?.toString() ?? '—',
      icon: Folder,
      color: 'text-blue-500',
    },
    {
      label: 'Molecules',
      value: stats?.molecules ?? '—',
      icon: FileText,
      color: 'text-cyan-500',
    },
    {
      label: 'Proteins',
      value: stats?.proteins ?? '—',
      icon: Database,
      color: 'text-emerald-500',
    },
    {
      label: 'Storage Used',
      value: stats?.storage ?? '—',
      icon: HardDrive,
      color: 'text-amber-500',
    },
  ];

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        {statCards.map((stat, i) => (
          <Card key={i}>
            <CardContent className="p-6">
              <div className="mb-2 flex items-start justify-between">
                <div className="text-muted-foreground text-sm font-medium">
                  {stat.label}
                </div>
                <div className={`text-lg ${stat.color}`}>
                  <stat.icon className="h-5 w-5" />
                </div>
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
          <SectionHeader
            title="Your Datasets"
            icon={<Folder className="text-primary h-5 w-5" />}
          />
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
                    <TableCell
                      colSpan={6}
                      className="text-muted-foreground text-center text-sm"
                    >
                      No datasets found.
                    </TableCell>
                  </TableRow>
                )}
                {datasets.map((ds, i) => (
                  <TableRow key={i}>
                    <TableCell className="font-medium">{ds.name}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Badge
                          variant={
                            ds.type === 'Molecules' ? 'default' : 'secondary'
                          }
                        >
                          {ds.type}
                        </Badge>
                      </div>
                    </TableCell>
                    <TableCell>{ds.count}</TableCell>
                    <TableCell>{ds.size}</TableCell>
                    <TableCell>{ds.updated}</TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-2">
                        <Button size="icon" variant="ghost" className="h-8 w-8">
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button size="icon" variant="ghost" className="h-8 w-8">
                          <Download className="h-4 w-4" />
                        </Button>
                        <Button
                          size="icon"
                          variant="ghost"
                          className="text-destructive hover:text-destructive h-8 w-8"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        </TabsContent>
        <TabsContent value="upload">
          <SectionHeader
            title="Upload New Data"
            icon={<Upload className="text-primary h-5 w-5" />}
          />
          <Card>
            <CardContent className="space-y-6 p-6">
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>Source</Label>
                    <Select
                      value={ingestSource}
                      onValueChange={setIngestSource}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select source" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="pubmed">PubMed (text)</SelectItem>
                        <SelectItem value="uniprot">
                          UniProt (protein)
                        </SelectItem>
                        <SelectItem value="chembl">
                          ChEMBL (molecule)
                        </SelectItem>
                        <SelectItem value="all">All sources</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Query</Label>
                    <Input
                      placeholder="EGFR, kinase inhibitor, BRCA1..."
                      value={ingestQuery}
                      onChange={(e) => setIngestQuery(e.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Limit</Label>
                    <Input
                      type="number"
                      min={1}
                      value={ingestLimit}
                      onChange={(e) =>
                        setIngestLimit(Number(e.target.value || 1))
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Collection (optional)</Label>
                    <Input
                      placeholder="bioflow_memory"
                      value={ingestCollection}
                      onChange={(e) => setIngestCollection(e.target.value)}
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <Input
                      id="sync-mode"
                      type="checkbox"
                      className="h-4 w-4"
                      checked={ingestSync}
                      onChange={(e) => setIngestSync(e.target.checked)}
                    />
                    <Label htmlFor="sync-mode">Run synchronously</Label>
                  </div>
                  <Button
                    onClick={startIngestion}
                    disabled={isSubmitting || !ingestQuery}
                  >
                    {isSubmitting ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : null}
                    Start Ingestion
                  </Button>
                  {ingestError && (
                    <div className="text-destructive text-sm">
                      {ingestError}
                    </div>
                  )}
                  {ingestMessage && (
                    <div className="text-sm text-emerald-600">
                      {ingestMessage}
                    </div>
                  )}
                </div>
                <div className="space-y-3">
                  <div className="flex flex-col items-center justify-center space-y-3 rounded-lg border-2 border-dashed p-8 text-center">
                    <div className="bg-primary/10 text-primary flex h-14 w-14 items-center justify-center rounded-full">
                      <CloudUpload className="h-7 w-7" />
                    </div>
                    <div className="text-base font-semibold">
                      Upload JSON batch
                    </div>
                    <div className="text-muted-foreground text-sm">
                      Array of records with <code>content</code>,{' '}
                      <code>modality</code>, and <code>metadata</code>.
                    </div>
                    <Button
                      variant="outline"
                      onClick={() => {
                        const input = document.createElement('input');
                        input.type = 'file';
                        input.accept = 'application/json';
                        input.onchange = (e) => {
                          const file = (e.target as HTMLInputElement)
                            .files?.[0];
                          if (file) handleBatchUpload(file);
                        };
                        input.click();
                      }}
                    >
                      Choose File
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="processing">
          <Card>
            <CardContent className="p-6">
              {jobs.length === 0 ? (
                <div className="text-muted-foreground text-center">
                  No active processing tasks.
                </div>
              ) : (
                <div className="space-y-3">
                  {jobs.map((job) => (
                    <div
                      key={job.job_id}
                      className="flex items-center justify-between rounded-lg border p-3"
                    >
                      <div>
                        <div className="font-medium">{job.job_id}</div>
                        <div className="text-muted-foreground text-xs">
                          {job.source || job.type || 'ingestion'}
                        </div>
                      </div>
                      <div className="text-sm">
                        <Badge
                          variant={
                            job.status === 'completed' ? 'default' : 'secondary'
                          }
                        >
                          {job.status || 'pending'}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <div id="analytics" className="space-y-4 pt-6">
        <SectionHeader
          title="Analytics"
          icon={<BarChart2 className="text-primary h-5 w-5" />}
        />
        <Card>
          <CardContent className="p-6">
            <div className="grid grid-cols-1 gap-4 text-sm md:grid-cols-3">
              <div className="rounded-lg border p-4">
                <div className="text-muted-foreground">Total Datasets</div>
                <div className="text-2xl font-semibold">
                  {stats?.datasets ?? '—'}
                </div>
              </div>
              <div className="rounded-lg border p-4">
                <div className="text-muted-foreground">Molecules Indexed</div>
                <div className="text-2xl font-semibold">
                  {stats?.molecules ?? '—'}
                </div>
              </div>
              <div className="rounded-lg border p-4">
                <div className="text-muted-foreground">Proteins Indexed</div>
                <div className="text-2xl font-semibold">
                  {stats?.proteins ?? '—'}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
