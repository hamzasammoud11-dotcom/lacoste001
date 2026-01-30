import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

/**
 * POST /api/search/image
 * 
 * Search for similar biological images.
 * 
 * Body: {
 *   image: string (file path, URL, or base64)
 *   image_type?: string
 *   collection?: string
 *   top_k?: number
 *   use_mmr?: boolean
 *   lambda_param?: number
 *   filters?: object
 * }
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const {
      image,
      image_type = 'other',
      collection,
      top_k = 10,
      use_mmr = true,
      lambda_param = 0.7,
      filters = {}
    } = body;

    if (!image) {
      return NextResponse.json(
        { error: "image is required for search (file path, URL, or base64)" },
        { status: 400 }
      );
    }

    // Call the FastAPI backend with JSON body
    const url = `${API_CONFIG.baseUrl}/api/search/image`;
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image,
        image_type,
        collection,
        top_k,
        use_mmr,
        lambda_param,
        filters,
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
    console.error("Image search API error:", error);
    return NextResponse.json(
      {
        results: [],
        query: "image_search",
        modality: "image",
        total_found: 0,
        returned: 0,
        error: `Image search failed: ${error instanceof Error ? error.message : 'Unknown error'}. Ensure backend is running and image modality is configured.`
      },
      { status: 503 }
    );
  }
}
