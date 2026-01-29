import { NextResponse } from 'next/server';

import { getProteins } from '@/lib/api';
import { mockProteins } from '@/lib/mock-proteins';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const backendProteins = await getProteins();
    
    // Filter proteins that have valid pdbIds (4-letter code)
    const proteinsWithPdb = backendProteins.filter(
      (p) => p.pdbId && p.pdbId.length === 4
    );
    
    // If we have proteins with PDB IDs from backend, use those
    // Otherwise, use mock proteins that have valid PDB structures
    const result = proteinsWithPdb.length > 0 ? proteinsWithPdb : mockProteins;
    
    return NextResponse.json(result);
  } catch (error) {
    console.warn('Proteins API error, using mock data:', error);
    // Return mock data as fallback
    return NextResponse.json(mockProteins);
  }
}
