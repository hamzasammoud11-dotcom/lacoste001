import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

/**
 * POST /api/neighbors
 * 
 * Navigate neighbors - guided exploration for Use Case 4.
 * Given an item, find semantically similar items across modalities
 * with controlled diversity for exploration.
 * 
 * Body: {
 *   item_id: string,
 *   collection?: string,
 *   top_k?: number,
 *   include_cross_modal?: boolean,
 *   diversity?: number (0-1)
 * }
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const {
      item_id,
      collection,
      top_k = 20,
      include_cross_modal = true,
      diversity = 0.3,
    } = body;

    if (!item_id) {
      return NextResponse.json(
        { error: "item_id is required" },
        { status: 400 }
      );
    }

    const response = await fetch(`${API_CONFIG.baseUrl}/api/neighbors`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        item_id,
        collection,
        top_k,
        include_cross_modal,
        diversity,
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
    console.error("Neighbors API error:", error);
    return NextResponse.json(
      {
        neighbors: [],
        error: `Neighbor search failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      },
      { status: 503 }
    );
  }
}
