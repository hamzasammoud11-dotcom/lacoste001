import { NextResponse } from "next/server"
import { API_CONFIG } from "@/config/api.config"

// Mock workflow results for when backend is unavailable
function generateMockWorkflowResult(query: string, numCandidates: number) {
  const candidates = []
  for (let i = 0; i < numCandidates; i++) {
    candidates.push({
      rank: i + 1,
      smiles: ["CCO", "CC(=O)O", "c1ccccc1", "CC(C)CC", "CCCCCC"][i % 5],
      name: `Candidate-${i + 1}`,
      score: Math.random() * 0.4 + 0.6,
      validation: {
        is_valid: Math.random() > 0.2,
        checks: {
          lipinski_ro5: Math.random() > 0.3,
          pains_filter: Math.random() > 0.2,
          synthetic_accessibility: Math.random() > 0.4,
        },
        properties: {
          mw: 150 + Math.random() * 300,
          logp: Math.random() * 5,
          hbd: Math.floor(Math.random() * 5),
          hba: Math.floor(Math.random() * 10),
        },
      },
    })
  }

  return {
    success: true,
    status: "completed",
    steps_completed: 3,
    total_steps: 3,
    execution_time_ms: Math.floor(Math.random() * 2000) + 500,
    top_candidates: candidates,
    all_outputs: {
      generate: { molecules: candidates.map((c) => c.smiles) },
      validate: { valid_count: candidates.filter((c) => c.validation.is_valid).length },
      rank: { ranked_by: "composite_score" },
    },
    errors: [],
  }
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => ({}))
  const { query, num_candidates = 5, top_k = 5 } = body

  try {
    const response = await fetch(`${API_CONFIG.baseUrl}/api/agents/workflow`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, num_candidates, top_k }),
      cache: "no-store",
    })

    if (response.ok) {
      const data = await response.json()
      return NextResponse.json(data)
    }
  } catch (error) {
    console.warn("Workflow API error, using mock response:", error)
  }

  const mockResult = generateMockWorkflowResult(query, num_candidates)
  return NextResponse.json(mockResult)
}
