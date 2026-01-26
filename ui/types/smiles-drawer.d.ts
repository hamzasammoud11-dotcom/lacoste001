declare module 'smiles-drawer' {
  interface SmiDrawerOptions {
    width?: number;
    height?: number;
    bondThickness?: number;
    bondLength?: number;
    shortBondLength?: number;
    bondSpacing?: number;
    atomVisualization?: 'default' | 'balls' | 'none';
    isomeric?: boolean;
    debug?: boolean;
    terminalCarbons?: boolean;
    explicitHydrogens?: boolean;
    compactDrawing?: boolean;
    fontSizeLarge?: number;
    fontSizeSmall?: number;
    padding?: number;
  }

  class SmiDrawer {
    constructor(options?: SmiDrawerOptions);
    draw(
      smiles: string,
      target: string | HTMLCanvasElement,
      theme?: 'light' | 'dark',
      onSuccess?: () => void,
      onError?: (error: Error) => void
    ): void;
  }

  export { SmiDrawer };
  export default { SmiDrawer };
}
