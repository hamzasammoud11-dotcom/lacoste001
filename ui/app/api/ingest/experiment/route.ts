import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

/**
 * POST /api/ingest/experiment
 * 
 * Ingest a single experimental result.
 * Supports Use Case 4: Measurements, conditions, outcomes.
 * 
 * Body: {
 *   title: string,
 *   type?: string,
 *   measurements?: Array<{name: string, value: number, unit: string}>,
 *   conditions?: object,
 *   outcome?: string,
 *   quality_score?: number,
 *   description?: string,
 *   molecule?: string,
 *   target?: string,
 *   collection?: string
 * }
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const {
      experiment_id,
      title,
      type = 'other',
      measurements = [],
      conditions = {},
      outcome = 'unknown',
      quality_score = 0.5,
      description = '',
      protocol,
      molecule,
      target,
      metadata,
      collection = 'bioflow_memory',
    } = body;

    if (!title) {
      return NextResponse.json(
        { error: "title is required" },
        { status: 400 }
      );
    }

    const response = await fetch(`${API_CONFIG.baseUrl}/api/ingest/experiment`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        experiment_id,
        title,
        type,
        measurements,
        conditions,
        outcome,
        quality_score,
        description,
        protocol,
        molecule,
        target,
        metadata,
        collection,
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
    console.error("Experiment ingestion API error:", error);
    return NextResponse.json(
      {
        success: false,
        error: `Experiment ingestion failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      },
      { status: 503 }
    );
  }
}
