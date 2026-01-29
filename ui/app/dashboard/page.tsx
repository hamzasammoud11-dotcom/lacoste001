'use client';

import {
  ArrowUp,
  Beaker,
  BookOpen,
  Database,
  Dna,
  FileText,
  Search,
  Sparkles,
  Zap,
} from 'lucide-react';
import Link from 'next/link';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';

export default function DashboardHome() {
  return (
    <div className="animate-in fade-in space-y-8 p-8 duration-500">
      <div className="flex flex-col gap-6 lg:flex-row">
        <div className="from-primary/10 via-background to-background flex-1 rounded-2xl border bg-linear-to-br p-8">
          <Badge variant="secondary" className="mb-4">
            New • BioFlow 2.0
          </Badge>
          <h1 className="mb-4 text-4xl font-bold tracking-tight">
            AI-Powered Drug Discovery
          </h1>
          <p className="text-muted-foreground mb-6 max-w-xl text-lg">
            Run discovery pipelines, predict binding, and surface evidence in
            one streamlined workspace.
          </p>
          <div className="mb-6 flex gap-2">
            <Badge
              variant="outline"
              className="bg-primary/5 border-primary/20 text-primary"
            >
              Model-aware search
            </Badge>
            <Badge
              variant="outline"
              className="border-green-500/20 bg-green-500/10 text-green-700 dark:text-green-400"
            >
              Evidence-linked
            </Badge>
            <Badge
              variant="outline"
              className="border-amber-500/20 bg-amber-500/10 text-amber-700 dark:text-amber-400"
            >
              Fast iteration
            </Badge>
          </div>

          <div className="flex gap-4">
            <Link href="/dashboard/discovery">
              <Button size="lg" className="font-semibold">
                Start Discovery
              </Button>
            </Link>
            <Link href="/dashboard/explorer">
              <Button size="lg" variant="outline">
                Explore Data
              </Button>
            </Link>
          </div>
        </div>

        <div className="lg:w-[350px]">
          <Card className="h-full">
            <CardContent className="flex h-full flex-col justify-between p-6">
              <div>
                <div className="text-muted-foreground mb-2 text-xs font-bold tracking-wider uppercase">
                  Today
                </div>
                <div className="mb-2 text-4xl font-bold">156 Discoveries</div>
                <div className="flex items-center gap-1 text-sm font-medium text-green-600">
                  <ArrowUp className="h-4 w-4" />
                  +12% vs last week
                </div>
              </div>

              <Separator className="my-4" />

              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="flex items-center gap-2">
                    <span className="bg-primary h-2 w-2 rounded-full"></span>
                    Discovery
                  </span>
                  <span className="font-mono font-medium">64</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-green-500"></span>
                    Prediction
                  </span>
                  <span className="font-mono font-medium">42</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-amber-500"></span>
                    Evidence
                  </span>
                  <span className="font-mono font-medium">50</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        {[
          {
            label: 'Molecules',
            value: '12.5M',
            icon: <Beaker className="h-5 w-5 text-blue-500" />,
            change: '+2.3%',
            color: 'text-blue-500',
          },
          {
            label: 'Proteins',
            value: '847K',
            icon: <Dna className="h-5 w-5 text-cyan-500" />,
            change: '+1.8%',
            color: 'text-cyan-500',
          },
          {
            label: 'Papers',
            value: '1.2M',
            icon: <BookOpen className="h-5 w-5 text-emerald-500" />,
            change: '+5.2%',
            color: 'text-emerald-500',
          },
          {
            label: 'Discoveries',
            value: '156',
            icon: <Sparkles className="h-5 w-5 text-amber-500" />,
            change: '+12%',
            color: 'text-amber-500',
          },
        ].map((metric, i) => (
          <Card key={i}>
            <CardContent className="p-6">
              <div className="mb-2 flex items-start justify-between">
                <div className="text-muted-foreground text-sm font-medium">
                  {metric.label}
                </div>
                <div className="text-lg">{metric.icon}</div>
              </div>
              <div className="mb-1 text-2xl font-bold">{metric.value}</div>
              <div className="flex items-center gap-1 text-xs font-medium text-green-500">
                <ArrowUp className="h-3 w-3" />
                {metric.change}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="pt-4">
        <div className="mb-4 flex items-center gap-2">
          <Zap className="h-5 w-5 text-amber-500" />
          <h2 className="text-xl font-semibold">Quick Actions</h2>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
          <Link href="/dashboard/molecules-2d" className="block">
            <Card className="hover:bg-accent/50 h-full cursor-pointer transition-colors">
              <CardContent className="flex flex-col items-center gap-3 p-6 text-center">
                <div className="bg-primary/10 text-primary flex h-10 w-10 items-center justify-center rounded-full">
                  <Search className="h-5 w-5" />
                </div>
                <div>
                  <div className="font-semibold">Molecules 2D</div>
                  <div className="text-muted-foreground text-sm">
                    View 2D structures
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>
          <Link href="/dashboard/molecules-3d" className="block">
            <Card className="hover:bg-accent/50 h-full cursor-pointer transition-colors">
              <CardContent className="flex flex-col items-center gap-3 p-6 text-center">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-500/10 text-blue-500">
                  <Database className="h-5 w-5" />
                </div>
                <div>
                  <div className="font-semibold">Molecules 3D</div>
                  <div className="text-muted-foreground text-sm">
                    View 3D structures
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>
          <Link href="/dashboard/proteins-3d" className="block">
            <Card className="hover:bg-accent/50 h-full cursor-pointer transition-colors">
              <CardContent className="flex flex-col items-center gap-3 p-6 text-center">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-purple-500/10 text-purple-500">
                  <FileText className="h-5 w-5" />
                </div>
                <div>
                  <div className="font-semibold">Proteins 3D</div>
                  <div className="text-muted-foreground text-sm">
                    View protein structures
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>
          <Link href="/dashboard/discovery" className="block">
            <Card className="hover:bg-accent/50 h-full cursor-pointer transition-colors">
              <CardContent className="flex flex-col items-center gap-3 p-6 text-center">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-500/10 text-slate-500">
                  <Sparkles className="h-5 w-5" />
                </div>
                <div>
                  <div className="font-semibold">Discovery</div>
                  <div className="text-muted-foreground text-sm">
                    Run predictions
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>
        </div>
      </div>
    </div>
  );
}
