import { NextResponse } from 'next/server';

import { API_CONFIG } from '@/config/api.config';

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({}));

  try {
    const response = await fetch(`${API_CONFIG.baseUrl}/api/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      cache: 'no-store',
    });

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
    console.warn('Search API error, using mock response:', error);
    return NextResponse.json({
      results: [],
      query: String(body?.query ?? ''),
      modality: 'auto',
      total_found: 0,
      returned: 0,
      diversity_score: null,
      filters_applied: body?.filters ?? {},
      search_time_ms: 0,
    });
  }
}
