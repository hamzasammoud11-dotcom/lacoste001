import { NextResponse } from 'next/server';

import { getStats } from '@/lib/api';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const data = await getStats();
    return NextResponse.json(data);
  } catch (err) {
    return NextResponse.json({ error: `Failed to fetch stats: ${err}` }, { status: 500 });
  }
}
