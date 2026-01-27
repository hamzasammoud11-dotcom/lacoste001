import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  try {
    const response = await fetch(`${API_CONFIG.baseUrl}/api/molecules/${id}`);
    
    if (!response.ok) {
      if (response.status === 404) {
        return NextResponse.json({ error: 'Molecule not found' }, { status: 404 });
      }
      throw new Error(`Backend returned ${response.status}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Molecule fetch error:', error);
    return NextResponse.json(
      { error: `Failed to fetch molecule: ${error instanceof Error ? error.message : 'Unknown error'}. Ensure backend is running.` },
      { status: 503 }
    );
  }
}
