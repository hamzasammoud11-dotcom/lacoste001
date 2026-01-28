import { NextResponse } from 'next/server';

import { API_CONFIG } from '@/config/api.config';

import { molecules } from '../_mock/molecules';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = searchParams.get('limit') || '20';
  const offset = searchParams.get('offset') || '0';

  try {
    const response = await fetch(
      `${API_CONFIG.baseUrl}/api/molecules?limit=${limit}&offset=${offset}`,
      { next: { revalidate: 60 } } // Cache for 60 seconds
    );
    
    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }
    
    const data = await response.json();
    const result = data.molecules || data;
    
    // If backend returns empty, use mock data
    if (!result || result.length === 0) {
      return NextResponse.json(molecules);
    }
    
    return NextResponse.json(result);
  } catch (error) {
    console.warn("Molecules API error, using mock data:", error);
    // Return mock data as fallback
    return NextResponse.json(molecules);
  }
}
