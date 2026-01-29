import { NextResponse } from 'next/server';

import { getProteins } from '@/lib/api';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const result = await getProteins();
    return NextResponse.json(result);
  } catch (error) {
    console.error('Proteins API error:', error);
    return NextResponse.json({ error: 'Failed to fetch proteins' }, { status: 500 });
  }
}
