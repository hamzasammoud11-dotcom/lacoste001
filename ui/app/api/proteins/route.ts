import { NextResponse } from 'next/server';

import { proteins } from '../_mock/proteins';

export async function GET() {
  return NextResponse.json(proteins);
}
