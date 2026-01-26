'use client';

import { useState } from 'react';

import { PageHeader } from '@/components/page-header';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ProteinViewer } from '@/components/visualization/protein-viewer';
import { Smiles2DViewer } from '@/components/visualization/smiles-2d-viewer';

// Example molecules
const EXAMPLE_SMILES = [
  { name: 'Aspirin', smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O' },
  { name: 'Caffeine', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' },
  { name: 'Ibuprofen', smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O' },
  { name: 'Paracetamol', smiles: 'CC(=O)NC1=CC=C(C=C1)O' },
];

// Example proteins (PDB IDs)
const EXAMPLE_PROTEINS = [
  { name: 'Hemoglobin', pdbId: '1HHO' },
  { name: 'Insulin', pdbId: '1ZNI' },
  { name: 'Lysozyme', pdbId: '1LYZ' },
  { name: 'Green Fluorescent Protein', pdbId: '1EMA' },
];

export default function VisualizationPage() {
  const [customSmiles, setCustomSmiles] = useState('');
  const [customPdbId, setCustomPdbId] = useState('');
  const [activeSmiles, setActiveSmiles] = useState(EXAMPLE_SMILES[0].smiles);
  const [activePdbId, setActivePdbId] = useState(EXAMPLE_PROTEINS[0].pdbId);

  return (
    <div className="container mx-auto p-6 space-y-6">
      <PageHeader
        title="Molecular Visualization"
        subtitle="View 2D molecule structures and 3D protein models"
        breadcrumbs={[
          { label: 'Home', href: '/' },
          { label: 'Visualization' },
        ]}
      />

      <Tabs defaultValue="molecules" className="space-y-6">
        <TabsList>
          <TabsTrigger value="molecules">2D Molecules</TabsTrigger>
          <TabsTrigger value="proteins">3D Proteins</TabsTrigger>
        </TabsList>

        <TabsContent value="molecules" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Controls */}
            <Card>
              <CardHeader>
                <CardTitle>Molecule Selection</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Example Molecules</Label>
                  <div className="flex flex-wrap gap-2">
                    {EXAMPLE_SMILES.map((mol) => (
                      <Button
                        key={mol.name}
                        variant={activeSmiles === mol.smiles ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setActiveSmiles(mol.smiles)}
                      >
                        {mol.name}
                      </Button>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="custom-smiles">Custom SMILES</Label>
                  <div className="flex gap-2">
                    <Input
                      id="custom-smiles"
                      placeholder="Enter SMILES string..."
                      value={customSmiles}
                      onChange={(e) => setCustomSmiles(e.target.value)}
                    />
                    <Button
                      onClick={() => customSmiles && setActiveSmiles(customSmiles)}
                      disabled={!customSmiles}
                    >
                      View
                    </Button>
                  </div>
                </div>

                <div className="p-3 bg-muted rounded-lg">
                  <Label className="text-xs text-muted-foreground">Current SMILES</Label>
                  <code className="block text-sm font-mono mt-1 break-all">
                    {activeSmiles}
                  </code>
                </div>
              </CardContent>
            </Card>

            {/* Viewer */}
            <Card>
              <CardHeader>
                <CardTitle>2D Structure</CardTitle>
              </CardHeader>
              <CardContent className="flex justify-center">
                <Smiles2DViewer
                  smiles={activeSmiles}
                  width={400}
                  height={300}
                />
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="proteins" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Controls */}
            <Card>
              <CardHeader>
                <CardTitle>Protein Selection</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Example Proteins</Label>
                  <div className="flex flex-wrap gap-2">
                    {EXAMPLE_PROTEINS.map((protein) => (
                      <Button
                        key={protein.pdbId}
                        variant={activePdbId === protein.pdbId ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setActivePdbId(protein.pdbId)}
                      >
                        {protein.name}
                      </Button>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="custom-pdb">Custom PDB ID</Label>
                  <div className="flex gap-2">
                    <Input
                      id="custom-pdb"
                      placeholder="Enter PDB ID (e.g., 1HHO)..."
                      value={customPdbId}
                      onChange={(e) => setCustomPdbId(e.target.value.toUpperCase())}
                      maxLength={4}
                    />
                    <Button
                      onClick={() => customPdbId && setActivePdbId(customPdbId)}
                      disabled={!customPdbId || customPdbId.length !== 4}
                    >
                      View
                    </Button>
                  </div>
                </div>

                <div className="p-3 bg-muted rounded-lg">
                  <Label className="text-xs text-muted-foreground">Current PDB ID</Label>
                  <code className="block text-sm font-mono mt-1">
                    {activePdbId}
                  </code>
                </div>
              </CardContent>
            </Card>

            {/* Viewer */}
            <Card>
              <CardHeader>
                <CardTitle>3D Structure</CardTitle>
              </CardHeader>
              <CardContent className="flex justify-center">
                <ProteinViewer
                  pdbId={activePdbId}
                  width={500}
                  height={400}
                />
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
