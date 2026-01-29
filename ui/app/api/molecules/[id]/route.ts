import { NextResponse } from 'next/server';

import { getMolecule } from '@/lib/api';

export const dynamic = 'force-dynamic';

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  try {
    const result = await getMolecule(id);
    return NextResponse.json(result);
  } catch (error) {
    console.error(`Molecule details API error for ${id}:`, error);
    return NextResponse.json({ error: 'Failed to fetch molecule details' }, { status: 500 });
  }
}
