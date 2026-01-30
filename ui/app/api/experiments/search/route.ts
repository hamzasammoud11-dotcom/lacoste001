import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

/**
 * POST /api/experiments/search
 * 
 * Search experimental results with outcome-based filtering.
 * For Use Case 4: Find experiments with specific outcomes, targets, or types.
 * 
 * Body: {
 *   query: string,
 *   experiment_type?: string,
 *   outcome?: string ('success' | 'failure' | 'partial'),
 *   target?: string,
 *   quality_min?: number (0-1),
 *   top_k?: number
 * }
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const {
      query,
      experiment_type,
      outcome,
      target,
      quality_min,
      top_k = 20,
    } = body;

    if (!query) {
      return NextResponse.json(
        { error: "query is required" },
        { status: 400 }
      );
    }

    const response = await fetch(`${API_CONFIG.baseUrl}/api/experiments/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        experiment_type,
        outcome,
        target,
        quality_min,
        top_k,
      }),
      cache: 'no-store',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Backend returned ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error("Experiment search API error:", error);
    return NextResponse.json(
      {
        experiments: [],
        error: `Experiment search failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      },
      { status: 503 }
    );
  }
}
