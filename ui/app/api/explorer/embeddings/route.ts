import { NextResponse } from 'next/server';

import { API_CONFIG } from '@/config/api.config';

export const dynamic = 'force-dynamic';

export async function GET(request: Request) {
  const url = new URL(request.url);
  const params = url.searchParams.toString();

  try {
    const response = await fetch(
      `${API_CONFIG.baseUrl}/api/explorer/embeddings?${params}`,
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
  } catch (error) {
    console.warn('Explorer embeddings API error:', error);
    return NextResponse.json(
      { error: 'Backend unavailable', points: [] },
      { status: 503 },
    );
  }
}
