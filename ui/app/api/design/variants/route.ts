import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

/**
 * POST /api/design/variants
 * 
 * Design assistance - propose close but diverse variants.
 * For Use Case 4: Given a reference, find similar items that
 * offer diverse design alternatives with justifications.
 * 
 * Body: {
 *   reference: string,
 *   modality?: string ('auto' | 'molecule' | 'protein' | 'text'),
 *   num_variants?: number,
 *   diversity?: number (0-1),
 *   constraints?: object
 * }
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const {
      reference,
      modality = 'auto',
      num_variants = 5,
      diversity = 0.5,
      constraints,
    } = body;

    if (!reference) {
      return NextResponse.json(
        { error: "reference is required" },
        { status: 400 }
      );
    }

    const response = await fetch(`${API_CONFIG.baseUrl}/api/design/variants`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        reference,
        modality,
        num_variants,
        diversity,
        constraints,
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
    console.error("Design variants API error:", error);
    return NextResponse.json(
      {
        variants: [],
        error: `Design variant suggestion failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      },
      { status: 503 }
    );
  }
}
