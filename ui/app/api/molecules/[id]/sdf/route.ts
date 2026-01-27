import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  try {
    // Get molecule info from backend
    const moleculeRes = await fetch(`${API_CONFIG.baseUrl}/api/molecules/${id}`);
    
    if (!moleculeRes.ok) {
      if (moleculeRes.status === 404) {
        return NextResponse.json({ error: 'Molecule not found' }, { status: 404 });
      }
      throw new Error(`Backend returned ${moleculeRes.status}`);
    }
    
    const molecule = await moleculeRes.json();
    
    if (!molecule.pubchemCid) {
      return NextResponse.json({ error: 'No PubChem CID available for this molecule' }, { status: 400 });
    }

    // Fetch 3D SDF from PubChem
    const pubchemUrl = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${molecule.pubchemCid}/record/SDF?record_type=3d`;

    const response = await fetch(pubchemUrl, {
      next: { revalidate: 3600 },
    });

    if (!response.ok) {
      // Try 2D SDF if 3D is not available
      const pubchemUrl2D = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${molecule.pubchemCid}/record/SDF`;
      const response2D = await fetch(pubchemUrl2D, {
        next: { revalidate: 3600 },
      });

      if (!response2D.ok) {
        return NextResponse.json(
          { error: 'Failed to fetch SDF from PubChem' },
          { status: 502 }
        );
      }

      const sdfText = await response2D.text();
      return new NextResponse(sdfText, {
        headers: {
          'Content-Type': 'chemical/x-mdl-sdfile',
          'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400',
        },
      });
    }

    const sdfText = await response.text();

    return new NextResponse(sdfText, {
      headers: {
        'Content-Type': 'chemical/x-mdl-sdfile',
        'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400',
      },
    });
  } catch (error) {
    console.error('Error fetching SDF:', error);
    return NextResponse.json(
      { error: `Failed to fetch SDF: ${error instanceof Error ? error.message : 'Unknown error'}` },
      { status: 500 }
    );
  }
}
