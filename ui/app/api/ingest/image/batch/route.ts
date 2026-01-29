import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

/**
 * POST /api/ingest/image/batch
 * 
 * Batch ingest multiple biological images.
 * 
 * Body: {
 *   images: Array<{
 *     image: string
 *     image_type: string
 *     experiment_id?: string
 *     description?: string
 *     caption?: string
 *     metadata?: object
 *   }>
 *   collection?: string
 * }
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { images, collection = "bioflow_memory" } = body;

    if (!images || !Array.isArray(images) || images.length === 0) {
      return NextResponse.json(
        { error: "images array is required and must not be empty" },
        { status: 400 }
      );
    }

    // Call the FastAPI backend
    const response = await fetch(`${API_CONFIG.baseUrl}/api/ingest/image/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        images,
        collection,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Backend returned ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error("Batch image ingestion API error:", error);
    return NextResponse.json(
      {
        error: `Batch image ingestion failed: ${error instanceof Error ? error.message : 'Unknown error'}. Ensure backend is running.`
      },
      { status: 503 }
    );
  }
}
