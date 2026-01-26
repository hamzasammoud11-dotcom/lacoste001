import { NextResponse } from 'next/server';

import { proteins } from '../../_mock/proteins';

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const protein = proteins.find((p) => p.id === id || p.pdbId === id);

  if (!protein) {
    return NextResponse.json({ error: 'Protein not found' }, { status: 404 });
  }

  return NextResponse.json(protein);
}
