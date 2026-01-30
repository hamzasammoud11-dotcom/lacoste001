/**
 * GET /api/images
 * Fetch experimental images (gels, microscopy) from the data corpus
 * 
 * Query params:
 *   type?: "gel" | "microscopy" | "all" (default: "all")
 *   limit?: number (default: 30)
 *   outcome?: string - Filter by experimental outcome
 *   cell_line?: string - Filter by cell line
 *   treatment?: string - Filter by treatment
 */

import { NextRequest, NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

export async function GET(request: NextRequest) {
    const searchParams = request.nextUrl.searchParams;
    const imageType = searchParams.get('type') || 'all';
    const limit = searchParams.get('limit') || '30';
    const outcome = searchParams.get('outcome') || '';
    const cellLine = searchParams.get('cell_line') || '';
    const treatment = searchParams.get('treatment') || '';

    try {
        // Build query params for backend
        const backendParams = new URLSearchParams({
            type: imageType,
            limit: limit,
        });
        
        if (outcome) backendParams.append('outcome', outcome);
        if (cellLine) backendParams.append('cell_line', cellLine);
        if (treatment) backendParams.append('treatment', treatment);
        
        // Call the backend's /api/images endpoint directly
        const response = await fetch(
            `${API_CONFIG.baseUrl}/api/images?${backendParams.toString()}`,
            {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
            }
        );

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Backend error:', response.status, errorText);
            throw new Error(`Backend returned ${response.status}: ${errorText}`);
        }

        const data = await response.json();
        
        return NextResponse.json({
            images: data.images || [],
            count: data.count || 0,
            type: data.type || imageType
        });
    } catch (error) {
        console.error('Error fetching images:', error);
        return NextResponse.json(
            { error: 'Failed to fetch images', images: [], count: 0 },
            { status: 500 }
        );
    }
}
