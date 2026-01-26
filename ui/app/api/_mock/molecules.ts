export type Molecule = {
  id: string;
  name: string;
  smiles: string;
  pubchemCid: number;
  description?: string;
};

export const molecules: Molecule[] = [
  {
    id: 'caffeine',
    name: 'Caffeine',
    smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    pubchemCid: 2519,
    description: 'Xanthine alkaloid found in coffee and tea',
  },
  {
    id: 'aspirin',
    name: 'Aspirin',
    smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O',
    pubchemCid: 2244,
    description: 'Acetylsalicylic acid - common pain reliever',
  },
  {
    id: 'ibuprofen',
    name: 'Ibuprofen',
    smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
    pubchemCid: 3672,
    description: 'Non-steroidal anti-inflammatory drug (NSAID)',
  },
];
