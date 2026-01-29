import { NextResponse } from 'next/server';

import { getProtein } from '@/lib/api';

export const dynamic = 'force-dynamic';

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  try {
    const result = await getProtein(id);
    return NextResponse.json(result);
  } catch (error) {
    console.error(`Protein details API error for ${id}:`, error);
    return NextResponse.json({ error: 'Failed to fetch protein details' }, { status: 500 });
  }
}
