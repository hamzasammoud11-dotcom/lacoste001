declare module 'smiles-drawer' {
  export interface SmiDrawerOptions {
    width?: number;
    height?: number;
    bondThickness?: number;
    bondLength?: number;
    shortBondLength?: number;
    bondSpacing?: number;
    atomVisualization?: string;
    isomeric?: boolean;
    debug?: boolean;
    terminalCarbons?: boolean;
    explicitHydrogens?: boolean;
    compactDrawing?: boolean;
    fontSizeLarge?: number;
    fontSizeSmall?: number;
    padding?: number;
  }

  export class SmiDrawer {
    constructor(options?: SmiDrawerOptions);
    draw(tree: unknown, canvas: HTMLCanvasElement | null, theme?: string): void;
  }

  export function parse(
    smiles: string,
    successCallback: (tree: unknown) => void,
    errorCallback: (error: Error) => void,
  ): void;

  const SmilesDrawer: {
    SmiDrawer: typeof SmiDrawer;
    parse: typeof parse;
  };

  export default SmilesDrawer;
}
