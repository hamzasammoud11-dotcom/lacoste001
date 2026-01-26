import { NextResponse } from 'next/server';

import { proteins } from '../../../_mock/proteins';

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const protein = proteins.find((p) => p.id === id || p.pdbId === id);

  if (!protein) {
    return NextResponse.json({ error: 'Protein not found' }, { status: 404 });
  }

  try {
    const rcsbUrl = `https://files.rcsb.org/download/${protein.pdbId}.pdb`;

    const response = await fetch(rcsbUrl, {
      next: { revalidate: 3600 }, // Cache for 1 hour
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: `Failed to fetch PDB from RCSB: ${response.statusText}` },
        { status: 502 }
      );
    }

    const pdbText = await response.text();

    return new NextResponse(pdbText, {
      headers: {
        'Content-Type': 'text/plain',
        'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400',
      },
    });
  } catch (error) {
    console.error('Error fetching PDB:', error);
    return NextResponse.json(
      { error: 'Failed to fetch PDB data' },
      { status: 500 }
    );
  }
}
