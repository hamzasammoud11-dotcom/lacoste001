import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  const body = await request.json();
  const { query } = body;

  console.info("Starting discovery for:", query);

  // Here you would typically call your Python backend
  // e.g., await fetch('http://localhost:8000/api/discovery', { ... })

  // Mock response
  return NextResponse.json({ 
    success: true,
    jobId: "job_" + Date.now(),
    status: "processing",
    message: "Pipeline started successfully"
  });
}
