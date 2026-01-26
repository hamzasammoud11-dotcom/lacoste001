import { NextResponse } from 'next/server';

import { molecules } from '../../../_mock/molecules';

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const molecule = molecules.find((m) => m.id === id);

  if (!molecule) {
    return NextResponse.json({ error: 'Molecule not found' }, { status: 404 });
  }

  try {
    const pubchemUrl = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${molecule.pubchemCid}/record/SDF?record_type=3d`;

    const response = await fetch(pubchemUrl, {
      next: { revalidate: 3600 }, // Cache for 1 hour
    });

    if (!response.ok) {
      // Try 2D SDF as fallback if 3D is not available
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
      { error: 'Failed to fetch SDF data' },
      { status: 500 }
    );
  }
}
