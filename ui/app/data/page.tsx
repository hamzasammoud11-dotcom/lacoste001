import { Suspense } from 'react'
import { getApiUrl } from "@/lib/api-utils"
import { DataView } from "./data-view"
import { PageHeader } from "@/components/page-header"
import { Database, Loader2 } from "lucide-react"
import { DataResponse } from "@/types/data"

async function getData() {
    try {
        const apiUrl = getApiUrl()
        const baseUrl = apiUrl || 'http://localhost:3000'
        
        const res = await fetch(`${baseUrl}/api/data`, {
            cache: 'no-store'
        })
        
        if (!res.ok) {
            throw new Error(`Failed to fetch data: ${res.status}`)
        }
        
        const json = await res.json() as DataResponse
        return json
    } catch (error) {
        console.error("Error fetching data:", error)
        return { datasets: [], stats: { datasets: 0, molecules: "0", proteins: "0", storage: "0" } }
    }
}

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
