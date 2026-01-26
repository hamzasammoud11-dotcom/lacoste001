import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

import { molecules } from '../_mock/molecules';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = searchParams.get('limit') || '20';
  const offset = searchParams.get('offset') || '0';

  try {
    // Try to fetch from FastAPI backend
    const response = await fetch(
      `${API_CONFIG.baseUrl}/api/molecules?limit=${limit}&offset=${offset}`
    );
    
    if (response.ok) {
      const data = await response.json();
      return NextResponse.json(data.molecules || data);
    }
  } catch (error) {
    console.warn("Backend unavailable, using mock data");
  }

  // Fallback to mock data
  return NextResponse.json(molecules);
}
