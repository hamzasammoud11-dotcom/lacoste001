export type Protein = {
  id: string;
  pdbId: string;
  name: string;
  description?: string;
};

export const proteins: Protein[] = [
  {
    id: '1CRN',
    pdbId: '1CRN',
    name: 'Crambin',
    description: 'Small hydrophobic protein from Crambe abyssinica seeds',
  },
  {
    id: '1UBQ',
    pdbId: '1UBQ',
    name: 'Ubiquitin',
    description: 'Regulatory protein found in most eukaryotic cells',
  },
  {
    id: '4HHB',
    pdbId: '4HHB',
    name: 'Hemoglobin',
    description: 'Iron-containing oxygen-transport protein in red blood cells',
  },
];
