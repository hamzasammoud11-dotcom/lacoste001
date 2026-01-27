import { NextResponse } from 'next/server';
import { API_CONFIG } from '@/config/api.config';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { drugSmiles, targetSequence } = body;

    if (!drugSmiles || !targetSequence) {
      return NextResponse.json(
        { error: "drugSmiles and targetSequence are required" },
        { status: 400 }
      );
    }

    // Call the FastAPI backend
    const response = await fetch(`${API_CONFIG.baseUrl}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        drug_smiles: drugSmiles,
        target_sequence: targetSequence,
      }),
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error("Predict API error:", error);
    return NextResponse.json(
      { error: `Prediction failed: ${error instanceof Error ? error.message : 'Unknown error'}. Ensure backend is running.` },
      { status: 503 }
    );
  }
}
