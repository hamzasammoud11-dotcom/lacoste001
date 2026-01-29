import { Database, Loader2 } from 'lucide-react';
import { Suspense } from 'react';

import { PageHeader } from '@/components/page-header';
import { getStats } from '@/lib/api';

import { DataView } from './data-view';

export const dynamic = 'force-dynamic';

export default async function DataPage() {
  const { datasets, stats } = await getStats();

  return (
    <div className="animate-in fade-in space-y-8 duration-500">
      <PageHeader
        title="Data Management"
        subtitle="Upload, manage, and organize your datasets"
        icon={<Database className="h-8 w-8" />}
      />

      <Suspense
        fallback={
          <div className="flex h-[400px] w-full items-center justify-center">
            <Loader2 className="text-muted-foreground h-8 w-8 animate-spin" />
          </div>
        }
      >
        <DataView datasets={datasets} stats={stats} />
      </Suspense>
    </div>
  );
}
