import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

/**
 * POST /api/ingest/image
 * 
 * Ingest a single biological image.
 * 
 * Body: {
 *   image: string (file path, URL, or base64)
 *   image_type: "microscopy" | "gel" | "spectra" | "xray" | "other"
 *   experiment_id?: string
 *   description?: string
 *   caption?: string
 *   metadata?: object
 *   collection?: string
 * }
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const {
      image,
      image_type = "other",
      experiment_id,
      description = "",
      caption = "",
      metadata = {},
      collection = "bioflow_memory"
    } = body;

    if (!image) {
      return NextResponse.json(
        { error: "image is required (file path, URL, or base64)" },
        { status: 400 }
      );
    }

    // Call the FastAPI backend
    const response = await fetch(`${API_CONFIG.baseUrl}/api/ingest/image`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image,
        image_type,
        experiment_id,
        description,
        caption,
        metadata,
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
    console.error("Image ingestion API error:", error);
    return NextResponse.json(
      {
        error: `Image ingestion failed: ${error instanceof Error ? error.message : 'Unknown error'}. Ensure backend is running and BiomedCLIP is available.`
      },
      { status: 503 }
    );
  }
}
