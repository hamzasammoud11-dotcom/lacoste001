/**
 * POST /api/search/gel-microscopy
 * 
 * Visual similarity search for biological images (Western blots, gels, microscopy)
 * Proxies to the backend /api/search/gel-microscopy endpoint
 * 
 * Use Case 4: Upload a biological image and find experiments with similar visual patterns
 */

import { NextRequest, NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        
        // Forward the request to the Python backend
        const response = await fetch(
            `${API_CONFIG.baseUrl}/api/search/gel-microscopy`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            }
        );

        if (!response.ok) {
            const errorText = await response.text();
            let errorData;
            try {
                errorData = JSON.parse(errorText);
            } catch {
                errorData = { detail: errorText };
            }
            
            console.error('Backend error:', response.status, errorData);
            
            return NextResponse.json(
                { 
                    error: errorData.detail || `Backend returned ${response.status}`,
                    results: [],
                    total_found: 0,
                    returned: 0,
                },
                { status: response.status }
            );
        }

        const data = await response.json();
        return NextResponse.json(data);
        
    } catch (error) {
        console.error('Gel/Microscopy search error:', error);
        
        // Return a proper error response instead of crashing
        return NextResponse.json(
            {
                error: error instanceof Error ? error.message : 'Search failed',
                results: [],
                query_image_type: null,
                total_found: 0,
                returned: 0,
                filters_applied: {},
                message: 'Failed to connect to search service. Make sure the API server is running.',
            },
            { status: 500 }
        );
    }
}
