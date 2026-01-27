import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';
import { proteins } from '../_mock/proteins';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = searchParams.get('limit') || '20';
  const offset = searchParams.get('offset') || '0';

  try {
    const response = await fetch(
      `${API_CONFIG.baseUrl}/api/proteins?limit=${limit}&offset=${offset}`,
      { next: { revalidate: 60 } } // Cache for 60 seconds
    );
    
    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data.proteins || data);
  } catch (error) {
    console.warn("Proteins API error, using mock data:", error);
    // Return mock data as fallback
    return NextResponse.json(proteins);
  }
}
