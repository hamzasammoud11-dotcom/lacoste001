import { NextResponse } from 'next/server';

import { getExplorerPoints } from '@/lib/api';

export const dynamic = 'force-dynamic';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);

  try {
    const data = await getExplorerPoints(
      searchParams.get('dataset') || undefined,
      searchParams.get('view') || undefined,
      searchParams.get('colorBy') || undefined,
    );
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Invalid parameters' }, { status: 400 });
  }
}
