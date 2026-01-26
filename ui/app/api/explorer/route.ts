import { NextResponse } from "next/server";
import { getExplorerPoints } from "@/lib/explorer-service";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  
  try {
    const data = await getExplorerPoints(
      searchParams.get("dataset") || undefined,
      searchParams.get("view") || undefined,
      searchParams.get("colorBy") || undefined
    );
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ error: "Invalid parameters" }, { status: 400 });
  }
}
