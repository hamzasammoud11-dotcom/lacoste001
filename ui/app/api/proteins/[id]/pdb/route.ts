import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  try {
    // Get protein info from backend
    const proteinRes = await fetch(`${API_CONFIG.baseUrl}/api/proteins/${id}`);
    
    if (!proteinRes.ok) {
      if (proteinRes.status === 404) {
        return NextResponse.json({ error: 'Protein not found' }, { status: 404 });
      }
      throw new Error(`Backend returned ${proteinRes.status}`);
    }
    
    const protein = await proteinRes.json();
    
    if (!protein.pdbId) {
      return NextResponse.json({ error: 'No PDB ID available for this protein' }, { status: 400 });
    }

    const rcsbUrl = `https://files.rcsb.org/download/${protein.pdbId}.pdb`;

    const response = await fetch(rcsbUrl, {
      next: { revalidate: 3600 },
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
      { error: `Failed to fetch PDB: ${error instanceof Error ? error.message : 'Unknown error'}` },
      { status: 500 }
    );
  }
}
