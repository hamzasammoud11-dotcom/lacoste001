import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

/**
 * POST /api/search/image
 * 
 * Search for similar biological images.
 * 
 * Body: {
 *   image: string (file path, URL, or base64)
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

    // Build query parameters
    const params = new URLSearchParams({
      image,
      top_k: String(top_k),
      use_mmr: String(use_mmr),
      lambda_param: String(lambda_param),
    });

    if (collection) {
      params.append('collection', collection);
    }

    // Call the FastAPI backend
    const url = `${API_CONFIG.baseUrl}/api/search/image?${params.toString()}`;
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ filters }),
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
