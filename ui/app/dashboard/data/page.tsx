import { Database, Loader2 } from "lucide-react"
import { Suspense } from 'react'

import { PageHeader } from "@/components/page-header"
import { getData } from "@/lib/data-service"

import { DataView } from "./data-view"

export default async function DataPage() {
    const { datasets, stats } = await getData()

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
             <PageHeader
                title="Data Management"
                subtitle="Upload, manage, and organize your datasets"
                icon={<Database className="h-8 w-8" />} 
            />

            <Suspense fallback={
                <div className="flex h-[400px] w-full items-center justify-center">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
            }>
                <DataView datasets={datasets} stats={stats} />
            </Suspense>
        </div>
    )
}
