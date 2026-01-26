import { NextResponse } from "next/server";

import { getData } from "@/lib/data-service";

export async function GET() {
  const data = await getData();
  return NextResponse.json(data);
}
