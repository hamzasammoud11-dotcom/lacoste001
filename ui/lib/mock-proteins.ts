import type { Protein } from '@/types/visualization';

/**
 * Mock proteins with real PDB IDs that can be visualized in 3D.
 * These are well-known proteins with available PDB structures.
 */
export const mockProteins: Protein[] = [
  {
    id: '1CRN',
    pdbId: '1CRN',
    name: 'Crambin',
    description: 'Small hydrophobic protein from Crambe abyssinica seeds - excellent for testing',
  },
  {
    id: '1UBQ',
    pdbId: '1UBQ',
    name: 'Ubiquitin',
    description: 'Regulatory protein involved in protein degradation pathways',
  },
  {
    id: '4HHB',
    pdbId: '4HHB',
    name: 'Hemoglobin',
    description: 'Iron-containing oxygen-transport metalloprotein in red blood cells',
  },
  {
    id: '1AKE',
    pdbId: '1AKE',
    name: 'Adenylate Kinase',
    description: 'Enzyme catalyzing interconversion of adenine nucleotides',
  },
  {
    id: '2PTC',
    pdbId: '2PTC',
    name: 'Trypsin',
    description: 'Serine protease enzyme that cleaves peptide chains',
  },
  {
    id: '1MBO',
    pdbId: '1MBO',
    name: 'Myoglobin',
    description: 'Iron and oxygen-binding protein found in muscle tissue',
  },
  {
    id: '3PQR',
    pdbId: '3PQR',
    name: 'Green Fluorescent Protein',
    description: 'Protein that exhibits bright green fluorescence when exposed to light',
  },
  {
    id: '1HHO',
    pdbId: '1HHO',
    name: 'Deoxyhemoglobin',
    description: 'Form of hemoglobin without bound oxygen',
  },
  {
    id: '2LZM',
    pdbId: '2LZM',
    name: 'Lysozyme',
    description: 'Antimicrobial enzyme that damages bacterial cell walls',
  },
  {
    id: '1IGT',
    pdbId: '1IGT',
    name: 'Immunoglobulin G1',
    description: 'Antibody protein involved in immune response',
  },
];
