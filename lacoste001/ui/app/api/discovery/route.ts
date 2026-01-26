import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

export async function POST(request: Request) {
  const body = await request.json();
  const { query, searchType = "similarity", database = "all", limit = 10 } = body;

  console.info("Starting discovery for:", query);

  try {
    // Call the FastAPI backend
    const response = await fetch(`${API_CONFIG.baseUrl}/api/discovery`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        search_type: searchType,
        database,
        limit,
      }),
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error("Discovery API error:", error);
    
    // Fallback to mock if backend is unavailable
    return NextResponse.json({ 
      success: true,
      job_id: "job_" + Date.now(),
      status: "pending",
      message: "Pipeline started (mock mode - backend unavailable)"
    });
  }
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const jobId = searchParams.get('jobId');

  if (!jobId) {
    return NextResponse.json({ error: "jobId required" }, { status: 400 });
  }

  try {
    const response = await fetch(`${API_CONFIG.baseUrl}/api/discovery/${jobId}`);
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ 
      job_id: jobId,
      status: "unknown",
      error: "Backend unavailable"
    });
  }
}
