// Molecule types
export type Molecule = {
  id: string;
  name: string;
  smiles: string;
  pubchemCid: number;
  description?: string;
};

// Protein types
export type Protein = {
  id: string;
  pdbId: string;
  name: string;
  description?: string;
};

// Viewer representation types
export type MoleculeRepresentation = 'stick' | 'sphere' | 'line' | 'cartoon';
export type ProteinRepresentation = 'cartoon' | 'ball-and-stick' | 'surface' | 'ribbon';
