import { NextRequest, NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

/**
 * Cross-modal search API route
 * 
 * Combines multiple query types (compound, sequence, text, image) to find
 * related experiments across the unified corpus.
 * 
 * Example: "Show me experiments that used THIS compound with THIS gel result"
 */
export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        
        const response = await fetch(
            `${API_CONFIG.baseUrl}/api/search/cross-modal`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(body),
            }
        );
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
            return NextResponse.json(
                { error: error.detail || `Backend error: ${response.status}` },
                { status: response.status }
            );
        }
        
        const data = await response.json();
        return NextResponse.json(data);
        
    } catch (error) {
        console.error('[Cross-Modal Search Route] Error:', error);
        return NextResponse.json(
            { 
                error: 'Cross-modal search failed',
                message: error instanceof Error ? error.message : 'Unknown error',
                results: [],
                total_found: 0,
                returned: 0
            },
            { status: 500 }
        );
    }
}
