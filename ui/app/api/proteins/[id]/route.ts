import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';
import { proteins } from '../../_mock/proteins';

export async function GET(_request: Request, { params }: { params: { id: string } }) {
  const { id } = params;
  try {
    const response = await fetch(`${API_CONFIG.baseUrl}/api/proteins/${id}`, { cache: 'no-store' });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      return NextResponse.json(
        { error: data?.detail || data?.error || `Backend returned ${response.status}` },
        { status: response.status }
      );
    }
    return NextResponse.json(data);
  } catch (error) {
    const fallback = proteins.find((p) => String(p.id) === String(id));
    if (fallback) {
      return NextResponse.json(fallback);
    }
    return NextResponse.json({ error: 'Backend unavailable' }, { status: 503 });
  }
}
