import { NextResponse } from 'next/server';

import { molecules } from '../../_mock/molecules';

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const molecule = molecules.find((m) => m.id === id);

  if (!molecule) {
    return NextResponse.json({ error: 'Molecule not found' }, { status: 404 });
  }

  return NextResponse.json(molecule);
}
