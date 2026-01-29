import { NextResponse } from 'next/server';

import { API_CONFIG } from '@/config/api.config';

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  try {
    const response = await fetch(
      `${API_CONFIG.baseUrl}/api/ingest/jobs/${id}`,
      {
        cache: 'no-store',
      },
    );
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      return NextResponse.json(
        {
          error:
            data?.detail ||
            data?.error ||
            `Backend returned ${response.status}`,
        },
        { status: response.status },
      );
    }
    return NextResponse.json(data);
  } catch (_error) {
    return NextResponse.json({ error: 'Backend unavailable' }, { status: 503 });
  }
}
