import type { Molecule, Protein } from './visualization-types';

const API_BASE = '/api';

// Generic fetch helper with error handling
async function fetchApi<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(error.error || `HTTP ${response.status}`);
  }
  return response.json();
}

async function fetchText(url: string): Promise<string> {
  const response = await fetch(url);
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(error.error || `HTTP ${response.status}`);
  }
  return response.text();
}

// Molecule API
export async function getMolecules(): Promise<Molecule[]> {
  return fetchApi<Molecule[]>(`${API_BASE}/molecules`);
}

export async function getMolecule(id: string): Promise<Molecule> {
  return fetchApi<Molecule>(`${API_BASE}/molecules/${id}`);
}

export async function getMoleculeSDF(id: string): Promise<string> {
  return fetchText(`${API_BASE}/molecules/${id}/sdf`);
}

// Protein API
export async function getProteins(): Promise<Protein[]> {
  return fetchApi<Protein[]>(`${API_BASE}/proteins`);
}

export async function getProtein(id: string): Promise<Protein> {
  return fetchApi<Protein>(`${API_BASE}/proteins/${id}`);
}

export async function getProteinPDB(id: string): Promise<string> {
  return fetchText(`${API_BASE}/proteins/${id}/pdb`);
}

// URL builders for direct links
export function getMoleculeSdfUrl(id: string): string {
  return `${API_BASE}/molecules/${id}/sdf`;
}

export function getProteinPdbUrl(pdbId: string): string {
  // Fetch directly from RCSB PDB for now
  return `https://files.rcsb.org/download/${pdbId.toUpperCase()}.pdb`;
}
