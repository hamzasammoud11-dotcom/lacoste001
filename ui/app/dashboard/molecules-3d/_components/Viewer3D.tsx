'use client';

import { useEffect, useRef, useState } from 'react';

import type { MoleculeRepresentation } from '@/types/visualization';

interface Viewer3DProps {
    sdfData: string;
    representation: MoleculeRepresentation;
    width: number;
    height: number;
    onReady: () => void;
    onError: (msg: string) => void;
}

export default function Viewer3D({
    sdfData,
    representation,
    width,
    height,
    onReady,
    onError,
}: Viewer3DProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const viewerRef = useRef<any>(null);
    const [isClient, setIsClient] = useState(false);

    // Ensure we're on client
    useEffect(() => {
        setIsClient(true);
    }, []);

    useEffect(() => {
        if (!isClient) return;
        
        let viewer: any;
        let mounted = true;

        async function init() {
            if (!containerRef.current || !mounted) return;

            try {
                // Dynamic import with error handling
                const $3Dmol = await import('3dmol').then(m => m.default).catch(() => null);
                
                if (!$3Dmol || !mounted) {
                    onError('3Dmol library failed to load');
                    return;
                }

                viewer = $3Dmol.createViewer(containerRef.current, {
                    backgroundColor: "#ffffff",
                });

                if (!mounted) return;
                
                viewerRef.current = viewer;

                viewer.addModel(sdfData, 'sdf');
                viewer.setStyle({}, getStyle(representation));
                viewer.zoomTo();
                viewer.render();

                onReady();
            } catch (err: any) {
                console.error('3Dmol init failed:', err);
                if (mounted) {
                    onError(
                        err?.message
                            ? `Visualization error: ${err.message}`
                            : 'Failed to load or initialize 3D viewer'
                    );
                }
            }
        }

        init();

        return () => {
            mounted = false;
            if (viewer?.removeAllModels) {
                try {
                    viewer.removeAllModels();
                } catch { }
            }
        };
    }, [sdfData, onReady, onError, isClient]);

    useEffect(() => {
        const v = viewerRef.current;
        if (!v) return;

        try {
            v.setStyle({}, getStyle(representation));
            v.render();
        } catch (err) {
            console.warn('Style update failed', err);
        }
    }, [representation]);

    return (
        <div
            ref={containerRef}
            style={{ width, height }}
            className="bg-background rounded-lg border"
        />
    );
}

function getStyle(rep: MoleculeRepresentation) {
    switch (rep) {
        case 'stick':
            return { stick: { radius: 0.15 } };
        case 'sphere':
            return { sphere: { scale: 0.3 } };
        case 'line':
            return { line: { linewidth: 2 } };
        case 'cartoon':
            return { cartoon: {} };
        default:
            return { stick: { radius: 0.15 } };
    }
}