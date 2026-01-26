import { NextResponse } from 'next/server';

import { molecules } from '../_mock/molecules';

export async function GET() {
  return NextResponse.json(molecules);
}
