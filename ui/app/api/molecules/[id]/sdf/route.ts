import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

// Mock PubChem CIDs for common molecules when backend is unavailable
const MOCK_PUBCHEM_CIDS: Record<string, number> = {
  caffeine: 2519,
  aspirin: 2244,
  ibuprofen: 3672,
  acetaminophen: 1983,
  morphine: 5288826,
  penicillin: 5904,
  insulin: 16129672,
  glucose: 5793,
  ethanol: 702,
  water: 962,
};

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const moleculeId = id.toLowerCase();

  let pubchemCid: number | null = null;

  try {
    // Try to get molecule info from backend
    const moleculeRes = await fetch(`${API_CONFIG.baseUrl}/api/molecules/${id}`);
    
    if (moleculeRes.ok) {
      const molecule = await moleculeRes.json();
      pubchemCid = molecule.pubchemCid;
    }
  } catch (error) {
    console.error('Error fetching SDF:', error);
    // Backend unavailable, use mock data
  }

  // Fallback to mock CID if backend didn't provide one
  if (!pubchemCid) {
    pubchemCid = MOCK_PUBCHEM_CIDS[moleculeId] || null;
  }

  if (!pubchemCid) {
    return NextResponse.json({ error: 'No PubChem CID available for this molecule' }, { status: 400 });
  }

  try {
    // Fetch 3D SDF from PubChem
    const pubchemUrl = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${pubchemCid}/record/SDF?record_type=3d`;

    const response = await fetch(pubchemUrl, {
      next: { revalidate: 3600 },
    });

    if (!response.ok) {
      // Try 2D SDF if 3D is not available
      const pubchemUrl2D = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${pubchemCid}/record/SDF`;
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
    console.error('Error fetching SDF from PubChem:', error);
    return NextResponse.json(
      { error: `Failed to fetch SDF: ${error instanceof Error ? error.message : 'Unknown error'}` },
      { status: 500 }
    );
  }
}
