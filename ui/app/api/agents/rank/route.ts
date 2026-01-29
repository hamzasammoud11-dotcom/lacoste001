import { NextResponse } from 'next/server';

import { API_CONFIG } from '@/config/api.config';

export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({}));

  try {
    const response = await fetch(`${API_CONFIG.baseUrl}/api/agents/rank`, {
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
    console.warn('Rank API error, using mock response:', error);
    return NextResponse.json({
      success: true,
      ranked: [],
      metadata: { mock: true },
    });
  }
}
