import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  try {
    const response = await fetch(`${API_CONFIG.baseUrl}/api/proteins/${id}/pdb`, { cache: 'no-store' });
    if (!response.ok) {
      const data = await response.json().catch(() => null);
      return NextResponse.json(
        { error: data?.detail || data?.error || `Backend returned ${response.status}` },
        { status: response.status }
      );
    }
    const text = await response.text();
    return new NextResponse(text, {
      status: 200,
      headers: { 'Content-Type': 'chemical/x-pdb' },
    });
  } catch (error) {
    return NextResponse.json({ error: 'Backend unavailable' }, { status: 503 });
  }
}
