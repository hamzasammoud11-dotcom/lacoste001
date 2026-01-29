import { NextResponse } from 'next/server';

import { API_CONFIG } from '@/config/api.config';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { content, modality = 'smiles', metadata = {} } = body;

    if (!content) {
      return NextResponse.json(
        { error: 'content is required' },
        { status: 400 },
      );
    }

    // Call the FastAPI backend
    const response = await fetch(`${API_CONFIG.baseUrl}/api/ingest`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        content,
        modality,
        metadata,
      }),
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Ingest API error:', error);
    return NextResponse.json(
      {
        error: `Ingestion failed: ${error instanceof Error ? error.message : 'Unknown error'}. Ensure backend is running.`,
      },
      { status: 503 },
    );
  }
}
