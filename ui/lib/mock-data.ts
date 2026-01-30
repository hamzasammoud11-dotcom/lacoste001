import { DataPoint, ExplorerResponse } from '@/schemas/explorer';
import { EmbeddingPoint } from '@/schemas/visualization';
import { SearchResult } from '@/schemas/search';

// Color palette for different categories
const colors = {
  high: '#22c55e',    // green - high affinity
  medium: '#f59e0b',  // amber - medium affinity
  low: '#ef4444',     // red - low affinity
  text: '#3b82f6',    // blue
  molecule: '#22c55e', // green
  protein: '#f59e0b',  // amber
  unknown: '#8b5cf6',  // purple
};

// Generate a random value within a range
const rand = (min: number, max: number) => Math.random() * (max - min) + min;

// Mock Explorer Points (Vector Space visualization)
export function generateMockExplorerPoints(count: number = 150): DataPoint[] {
  const drugNames = [
    'Aspirin', 'Ibuprofen', 'Acetaminophen', 'Caffeine', 'Metformin',
    'Omeprazole', 'Lisinopril', 'Atorvastatin', 'Amlodipine', 'Metoprolol',
    'Simvastatin', 'Losartan', 'Gabapentin', 'Hydrochlorothiazide', 'Sertraline',
    'Fluoxetine', 'Citalopram', 'Escitalopram', 'Alprazolam', 'Clonazepam',
    'Diazepam', 'Lorazepam', 'Zolpidem', 'Trazodone', 'Quetiapine',
    'Aripiprazole', 'Risperidone', 'Olanzapine', 'Lamotrigine', 'Carbamazepine',
    'Levetiracetam', 'Topiramate', 'Pregabalin', 'Duloxetine', 'Venlafaxine',
    'Bupropion', 'Mirtazapine', 'Buspirone', 'Hydroxyzine', 'Propranolol',
    'Celecoxib', 'Naproxen', 'Meloxicam', 'Diclofenac', 'Indomethacin',
    'Piroxicam', 'Ketorolac', 'Tramadol', 'Codeine', 'Morphine',
  ];

  const proteins = [
    'ACE2', 'EGFR', 'BRAF', 'HER2', 'VEGFR', 'CDK4', 'CDK6', 'mTOR',
    'PI3K', 'AKT', 'MEK', 'ERK', 'JAK2', 'STAT3', 'BCL2', 'MCL1',
    'MDM2', 'p53', 'RAS', 'RAF', 'PARP', 'ALK', 'ROS1', 'MET',
  ];

  const points: DataPoint[] = [];

  // Generate clustered data points
  const clusters = [
    { centerX: -2, centerY: 2, centerZ: 1, affinityRange: [7, 9] },    // High affinity cluster
    { centerX: 1, centerY: -1, centerZ: -1, affinityRange: [5, 7] },   // Medium affinity cluster
    { centerX: -1, centerY: -2, centerZ: 2, affinityRange: [3, 5] },   // Low affinity cluster
    { centerX: 2, centerY: 1, centerZ: 0, affinityRange: [6, 8] },     // Mixed cluster
    { centerX: 0, centerY: 0, centerZ: -2, affinityRange: [4, 6] },    // Central cluster
  ];

  for (let i = 0; i < count; i++) {
    const cluster = clusters[i % clusters.length];
    const affinity = rand(cluster.affinityRange[0], cluster.affinityRange[1]);
    
    let color: string;
    if (affinity >= 7) {
      color = colors.high;
    } else if (affinity >= 5) {
      color = colors.medium;
    } else {
      color = colors.low;
    }

    const isProtein = Math.random() > 0.7;
    const nameList = isProtein ? proteins : drugNames;
    const baseName = nameList[i % nameList.length];
    const name = isProtein 
      ? `${baseName}-${Math.floor(rand(1, 100))}`
      : `${baseName} (${['oral', 'IV', 'topical'][Math.floor(rand(0, 3))]})`;

    points.push({
      x: cluster.centerX + rand(-1.5, 1.5),
      y: cluster.centerY + rand(-1.5, 1.5),
      z: cluster.centerZ + rand(-1.5, 1.5),
      color,
      name,
      affinity: parseFloat(affinity.toFixed(2)),
    });
  }

  return points;
}

export function getMockExplorerResponse(): ExplorerResponse {
  return {
    points: generateMockExplorerPoints(150),
    metrics: {
      activeMolecules: 1247,
      clusters: 5,
      avgConfidence: 0.847,
    },
  };
}

// Mock 3D Embedding Points for Visualization page
export function generateMock3DEmbeddingPoints(count: number = 80): EmbeddingPoint[] {
  const molecules = [
    { name: 'Aspirin', smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O' },
    { name: 'Caffeine', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' },
    { name: 'Ibuprofen', smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O' },
    { name: 'Paracetamol', smiles: 'CC(=O)NC1=CC=C(C=C1)O' },
    { name: 'Metformin', smiles: 'CN(C)C(=N)NC(=N)N' },
    { name: 'Warfarin', smiles: 'CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O' },
    { name: 'Omeprazole', smiles: 'COC1=CC2=C(C=C1)N=C(N2)S(=O)CC3=NC=C(C=C3C)OC' },
    { name: 'Atorvastatin', smiles: 'CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4' },
  ];

  const proteins = [
    { name: 'Human Serum Albumin', id: 'P02768' },
    { name: 'Cytochrome P450 3A4', id: 'P08684' },
    { name: 'Cyclooxygenase-2', id: 'P35354' },
    { name: 'Thrombin', id: 'P00734' },
    { name: 'Carbonic Anhydrase II', id: 'P00918' },
    { name: 'Beta-2 Adrenergic Receptor', id: 'P07550' },
    { name: 'Angiotensin-converting enzyme', id: 'P12821' },
    { name: 'HMG-CoA Reductase', id: 'P04035' },
  ];

  const textSources = [
    'Recent studies show promising results in binding affinity prediction using deep learning models.',
    'The molecular dynamics simulation revealed key interaction sites at the active pocket.',
    'Phase III clinical trial results demonstrate significant efficacy improvement.',
    'Structural analysis indicates hydrogen bonding with key residues Asp-124 and Glu-256.',
    'ADMET predictions suggest favorable pharmacokinetic properties.',
    'Molecular docking scores correlate well with experimental IC50 values.',
    'The compound shows selectivity for the target over related kinases.',
    'Crystallographic data reveals conformational changes upon ligand binding.',
  ];

  const points: EmbeddingPoint[] = [];

  // Generate molecule embeddings (cluster 1)
  for (let i = 0; i < Math.floor(count * 0.4); i++) {
    const mol = molecules[i % molecules.length];
    points.push({
      id: `mol-${i}`,
      x: rand(-2, 0) + rand(-0.5, 0.5),
      y: rand(0, 2) + rand(-0.5, 0.5),
      z: rand(-1, 1) + rand(-0.5, 0.5),
      label: mol.name,
      content: mol.smiles,
      modality: 'molecule',
      source: 'DrugBank',
      score: rand(0.7, 0.99),
      metadata: { mw: rand(150, 600), logP: rand(-1, 5) },
    });
  }

  // Generate protein embeddings (cluster 2)
  for (let i = 0; i < Math.floor(count * 0.3); i++) {
    const protein = proteins[i % proteins.length];
    points.push({
      id: `prot-${i}`,
      x: rand(0, 2) + rand(-0.5, 0.5),
      y: rand(-2, 0) + rand(-0.5, 0.5),
      z: rand(0, 2) + rand(-0.5, 0.5),
      label: protein.name,
      content: `UniProt: ${protein.id}`,
      modality: 'protein',
      source: 'UniProt',
      score: rand(0.6, 0.95),
      metadata: { uniprot_id: protein.id },
    });
  }

  // Generate text embeddings (cluster 3)
  for (let i = 0; i < Math.floor(count * 0.3); i++) {
    const text = textSources[i % textSources.length];
    points.push({
      id: `text-${i}`,
      x: rand(-1, 1) + rand(-0.5, 0.5),
      y: rand(-1, 1) + rand(-0.5, 0.5),
      z: rand(-2, 0) + rand(-0.5, 0.5),
      label: text.slice(0, 50) + '...',
      content: text,
      modality: 'text',
      source: 'PubMed',
      score: rand(0.5, 0.9),
      metadata: { pmid: `${Math.floor(rand(20000000, 40000000))}` },
    });
  }

  return points;
}

// Mock Search Results for 3D Visualization
export function generateMockSearchResults(count: number = 20): SearchResult[] {
  const results: SearchResult[] = [];

  const contents = [
    { modality: 'molecule', content: 'CC(=O)OC1=CC=CC=C1C(=O)O', source: 'DrugBank', citation: 'DrugBank DB00945' },
    { modality: 'molecule', content: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', source: 'ChEMBL', citation: 'ChEMBL113' },
    { modality: 'protein', content: 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH', source: 'UniProt', citation: 'P69905 - Hemoglobin subunit alpha' },
    { modality: 'protein', content: 'MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL', source: 'UniProt', citation: 'P00533 - EGFR' },
    { modality: 'text', content: 'The binding affinity of aspirin to COX-2 was measured using surface plasmon resonance (SPR) and showed a Kd of 1.2 μM.', source: 'PubMed', citation: 'PMID: 25678901' },
    { modality: 'text', content: 'Molecular dynamics simulations revealed that caffeine interacts primarily with adenosine receptors through π-stacking interactions.', source: 'PubMed', citation: 'PMID: 28901234' },
    { modality: 'text', content: 'High-throughput screening identified several lead compounds with sub-nanomolar potency against the target kinase.', source: 'bioRxiv', citation: 'doi:10.1101/2024.01.15.575432' },
    { modality: 'molecule', content: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', source: 'PubChem', citation: 'CID 3672' },
  ];

  for (let i = 0; i < count; i++) {
    const base = contents[i % contents.length];
    results.push({
      id: `result-${i}`,
      content: base.content,
      score: parseFloat(rand(0.5, 0.99).toFixed(3)),
      modality: base.modality,
      source: base.source,
      citation: base.citation,
      evidence_links: [
        { source: base.source, identifier: `ID-${i}`, url: `https://${base.source.toLowerCase()}.org/${i}`, label: `View on ${base.source}` },
      ],
    });
  }

  return results.sort((a, b) => b.score - a.score);
}
