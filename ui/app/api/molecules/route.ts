import { NextResponse } from 'next/server';

import { getMolecules } from '@/lib/api';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const result = await getMolecules();
    return NextResponse.json(result);
  } catch (error) {
    console.error('Molecules API error:', error);
    return NextResponse.json({ error: 'Failed to fetch molecules' }, { status: 500 });
  }
}
