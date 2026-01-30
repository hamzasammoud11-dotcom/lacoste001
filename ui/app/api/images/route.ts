/**
 * GET /api/images
 * Fetch experimental images (gels, microscopy) from the data corpus
 * 
 * Query params:
 *   type?: "gel" | "microscopy" | "spectra" | "all" (default: "all")
 *   limit?: number (default: 20)
 */

import { NextRequest, NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

export async function GET(request: NextRequest) {
    const searchParams = request.nextUrl.searchParams;
    const imageType = searchParams.get('type') || 'all';
    const limit = parseInt(searchParams.get('limit') || '20');

    try {
        // Search the vector database for images of the specified type
        const response = await fetch(`${API_CONFIG.baseUrl}/api/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: imageType === 'all' 
                    ? 'experimental images gel microscopy western blot fluorescence' 
                    : imageType === 'gel' 
                        ? 'western blot gel electrophoresis SDS-PAGE protein expression'
                        : imageType === 'microscopy'
                            ? 'microscopy fluorescence cell imaging confocal brightfield'
                            : imageType,
                type: 'text',
                limit: limit,
                filter: {
                    modality: 'image'
                }
            }),
        });

        if (!response.ok) {
            throw new Error(`Backend returned ${response.status}`);
        }

        const data = await response.json();
        
        // Filter results to only include images with the correct type
        const filteredResults = data.results?.filter((r: any) => {
            if (imageType === 'all') {
                return r.modality === 'image' && ['gel', 'microscopy', 'fluorescence'].includes(r.metadata?.image_type);
            }
            if (imageType === 'gel') {
                return r.modality === 'image' && r.metadata?.image_type === 'gel';
            }
            if (imageType === 'microscopy') {
                return r.modality === 'image' && ['microscopy', 'fluorescence'].includes(r.metadata?.image_type);
            }
            return r.modality === 'image';
        }) || [];

        return NextResponse.json({
            images: filteredResults,
            count: filteredResults.length,
            type: imageType
        });
    } catch (error) {
        console.error('Error fetching images:', error);
        return NextResponse.json(
            { error: 'Failed to fetch images', images: [], count: 0 },
            { status: 500 }
        );
    }
}
