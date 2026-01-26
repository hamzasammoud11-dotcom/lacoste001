"use client"

import { ArrowUp, Database, FileText, Search, Sparkles, Zap, Beaker, Dna, BookOpen } from "lucide-react"
import Link from "next/link"

import { SectionHeader } from "@/components/page-header"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"

export default function Home() {
  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      {/* Hero Section */}
      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-1 rounded-2xl bg-gradient-to-br from-primary/10 via-background to-background p-8 border">
          <Badge variant="secondary" className="mb-4">New • BioFlow 2.0</Badge>
          <h1 className="text-4xl font-bold tracking-tight mb-4">AI-Powered Drug Discovery</h1>
          <p className="text-lg text-muted-foreground mb-6 max-w-xl">
            Run discovery pipelines, predict binding, and surface evidence in one streamlined workspace.
          </p>
          <div className="flex gap-2 mb-6">
            <Badge variant="outline" className="bg-primary/5 border-primary/20 text-primary">Model-aware search</Badge>
            <Badge variant="outline" className="bg-green-500/10 border-green-500/20 text-green-700 dark:text-green-400">Evidence-linked</Badge>
            <Badge variant="outline" className="bg-amber-500/10 border-amber-500/20 text-amber-700 dark:text-amber-400">Fast iteration</Badge>
          </div>
          
          <div className="flex gap-4">
            <Link href="/discovery">
                <Button size="lg" className="font-semibold">
                    Start Discovery
                </Button>
            </Link>
            <Link href="/explorer">
                <Button size="lg" variant="outline">
                    Explore Data
                </Button>
            </Link>
          </div>
        </div>

        <div className="lg:w-[350px]">
          <Card className="h-full">
            <CardContent className="p-6 flex flex-col justify-between h-full">
                <div>
                   <div className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-2">Today</div>
                   <div className="text-4xl font-bold mb-2">156 Discoveries</div>
                   <div className="text-sm text-green-600 font-medium flex items-center gap-1">
                     <ArrowUp className="h-4 w-4" />
                     +12% vs last week
                   </div>
                </div>
                
                <Separator className="my-4" />

                <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                        <span className="flex items-center gap-2">
                             <span className="h-2 w-2 rounded-full bg-primary"></span>
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

      {/* Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
            { label: "Molecules", value: "12.5M", icon: <Beaker className="h-5 w-5 text-blue-500" />, change: "+2.3%" },
            { label: "Proteins", value: "847K", icon: <Dna className="h-5 w-5 text-cyan-500" />, change: "+1.8%" },
            { label: "Papers", value: "1.2M", icon: <BookOpen className="h-5 w-5 text-emerald-500" />, change: "+5.2%" },
            { label: "Discoveries", value: "156", icon: <Sparkles className="h-5 w-5 text-amber-500" />, change: "+12%" }
        ].map((metric, i) => (
            <Card key={i}>
                <CardContent className="p-6">
                    <div className="flex justify-between items-start mb-2">
                        <div className="text-sm font-medium text-muted-foreground">{metric.label}</div>
                        <div className="text-lg">{metric.icon}</div>
                    </div>
                    <div className="text-2xl font-bold mb-1">{metric.value}</div>
                    <div className="text-xs font-medium flex items-center gap-1 text-green-500">
                        <ArrowUp className="h-3 w-3" />
                        {metric.change}
                    </div>
                </CardContent>
            </Card>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="pt-4">
        <SectionHeader title="Quick Actions" icon={<Zap className="h-5 w-5 text-amber-500" />} />
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Link href="/discovery" className="block">
                <Card className="hover:bg-accent/50 transition-colors cursor-pointer h-full">
                    <CardContent className="p-6 flex flex-col items-center text-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                            <Search className="h-5 w-5" />
                        </div>
                        <div>
                            <div className="font-semibold">New Discovery</div>
                            <div className="text-sm text-muted-foreground">Start a pipeline</div>
                        </div>
                    </CardContent>
                </Card>
            </Link>
             <Link href="/explorer" className="block">
                <Card className="hover:bg-accent/50 transition-colors cursor-pointer h-full">
                    <CardContent className="p-6 flex flex-col items-center text-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-blue-500/10 flex items-center justify-center text-blue-500">
                            <Database className="h-5 w-5" />
                        </div>
                        <div>
                            <div className="font-semibold">Browse Data</div>
                            <div className="text-sm text-muted-foreground">Explore datasets</div>
                        </div>
                    </CardContent>
                </Card>
            </Link>
             <Link href="/data" className="block">
                <Card className="hover:bg-accent/50 transition-colors cursor-pointer h-full">
                    <CardContent className="p-6 flex flex-col items-center text-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-purple-500/10 flex items-center justify-center text-purple-500">
                            <FileText className="h-5 w-5" />
                        </div>
                        <div>
                            <div className="font-semibold">Training</div>
                            <div className="text-sm text-muted-foreground">Train new models</div>
                        </div>
                    </CardContent>
                </Card>
            </Link>
             <Link href="/settings" className="block">
                <Card className="hover:bg-accent/50 transition-colors cursor-pointer h-full">
                    <CardContent className="p-6 flex flex-col items-center text-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-slate-500/10 flex items-center justify-center text-slate-500">
                            <Sparkles className="h-5 w-5" />
                        </div>
                        <div>
                            <div className="font-semibold">View Insights</div>
                            <div className="text-sm text-muted-foreground">Check predictions</div>
                        </div>
                    </CardContent>
                </Card>
            </Link>
        </div>
      </div>
    </div>
  )
}
