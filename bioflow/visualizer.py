"""
BioFlow Visualizer - Embedding and Structure Visualization
===========================================================

This module provides visualization utilities for embeddings,
molecular structures, and search results.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Optional imports for visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class EmbeddingVisualizer:
    """Visualize high-dimensional embeddings in 2D/3D."""
    
    @staticmethod
    def reduce_dimensions(
        embeddings: np.ndarray,
        method: str = "pca",
        n_components: int = 2,
        **kwargs
    ) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization.
        
        Args:
            embeddings: Array of shape (n_samples, n_features).
            method: 'pca' or 'tsne'.
            n_components: Target dimensions (2 or 3).
            
        Returns:
            Reduced embeddings of shape (n_samples, n_components).
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for dimensionality reduction")
        
        if method == "pca":
            reducer = PCA(n_components=n_components, **kwargs)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return reducer.fit_transform(embeddings)
    
    @staticmethod
    def plot_embeddings_2d(
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        title: str = "Embedding Space",
        hover_data: Optional[List[Dict]] = None
    ):
        """
        Create 2D scatter plot of embeddings.
        
        Returns:
            Plotly figure object.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualization")
        
        if embeddings.shape[1] > 2:
            coords = EmbeddingVisualizer.reduce_dimensions(embeddings, "pca", 2)
        else:
            coords = embeddings
        
        fig = go.Figure()
        
        # Group by color if provided
        if colors:
            unique_colors = list(set(colors))
            for color in unique_colors:
                mask = [c == color for c in colors]
                x = [coords[i, 0] for i in range(len(coords)) if mask[i]]
                y = [coords[i, 1] for i in range(len(coords)) if mask[i]]
                text = [labels[i] if labels else f"Point {i}" for i in range(len(coords)) if mask[i]]
                
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    name=color,
                    text=text,
                    hoverinfo='text'
                ))
        else:
            fig.add_trace(go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode='markers',
                text=labels or [f"Point {i}" for i in range(len(coords))],
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            hovermode='closest'
        )
        
        return fig
    
    @staticmethod
    def plot_embeddings_3d(
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        title: str = "3D Embedding Space"
    ):
        """Create 3D scatter plot of embeddings."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualization")
        
        if embeddings.shape[1] > 3:
            coords = EmbeddingVisualizer.reduce_dimensions(embeddings, "pca", 3)
        else:
            coords = embeddings
        
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            text=labels or [f"Point {i}" for i in range(len(coords))],
            marker=dict(
                size=5,
                color=list(range(len(coords))) if not colors else None,
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Dim 1',
                yaxis_title='Dim 2',
                zaxis_title='Dim 3'
            )
        )
        
        return fig
    
    @staticmethod
    def plot_similarity_matrix(
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Similarity Matrix"
    ):
        """Plot pairwise similarity matrix."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualization")
        
        # Compute cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.clip(norms, 1e-9, None)
        similarity = np.dot(normalized, normalized.T)
        
        labels = labels or [f"Item {i}" for i in range(len(embeddings))]
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity,
            x=labels,
            y=labels,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Items",
            yaxis_title="Items"
        )
        
        return fig


class MoleculeVisualizer:
    """Visualize molecular structures."""
    
    @staticmethod
    def smiles_to_svg(smiles: str, size: Tuple[int, int] = (300, 200)) -> str:
        """
        Convert SMILES to SVG image.
        
        Args:
            smiles: SMILES string.
            size: (width, height) tuple.
            
        Returns:
            SVG string.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return f"<svg><text>Invalid SMILES</text></svg>"
            
            drawer = Draw.MolDraw2DSVG(size[0], size[1])
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            return drawer.GetDrawingText()
        except ImportError:
            return f"<svg><text>RDKit not available</text></svg>"
    
    @staticmethod
    def plot_molecule_grid(
        smiles_list: List[str],
        labels: Optional[List[str]] = None,
        mols_per_row: int = 4,
        size: Tuple[int, int] = (200, 200)
    ):
        """
        Create a grid of molecule images.
        
        Returns:
            PIL Image or Plotly figure.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            
            mols = [Chem.MolFromSmiles(s) for s in smiles_list]
            legends = labels or smiles_list
            
            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=mols_per_row,
                subImgSize=size,
                legends=legends
            )
            return img
        except ImportError:
            logger.warning("RDKit not available for molecule visualization")
            return None


class ResultsVisualizer:
    """Visualize search and pipeline results."""
    
    @staticmethod
    def plot_search_scores(
        results: List[Dict[str, Any]],
        title: str = "Search Results Scores"
    ):
        """Plot bar chart of search result scores."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required")
        
        labels = [r.get("content", "")[:30] for r in results]
        scores = [r.get("score", 0) for r in results]
        modalities = [r.get("modality", "unknown") for r in results]
        
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=scores,
            marker_color=[
                'blue' if m == 'text' else 'green' if m in ['smiles', 'molecule'] else 'red'
                for m in modalities
            ],
            text=[f"{m}: {s:.3f}" for m, s in zip(modalities, scores)],
            textposition='outside'
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title="Content",
            yaxis_title="Similarity Score",
            xaxis_tickangle=-45
        )
        
        return fig
    
    @staticmethod
    def plot_modality_distribution(
        items: List[Dict[str, Any]],
        title: str = "Modality Distribution"
    ):
        """Plot pie chart of modality distribution."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required")
        
        modalities = [item.get("modality", "unknown") for item in items]
        unique = list(set(modalities))
        counts = [modalities.count(m) for m in unique]
        
        fig = go.Figure(data=[go.Pie(
            labels=unique,
            values=counts,
            hole=0.3
        )])
        
        fig.update_layout(title=title)
        return fig
    
    @staticmethod
    def create_dashboard(
        search_results: List[Dict],
        embeddings: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None
    ):
        """Create a multi-panel dashboard."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Search Scores",
                "Modality Distribution",
                "Embedding Space",
                "Similarity Heatmap"
            ),
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ]
        )
        
        # Panel 1: Search scores
        scores = [r.get("score", 0) for r in search_results]
        fig.add_trace(
            go.Bar(y=scores, name="Scores"),
            row=1, col=1
        )
        
        # Panel 2: Modality distribution
        modalities = [r.get("modality", "unknown") for r in search_results]
        unique_mods = list(set(modalities))
        counts = [modalities.count(m) for m in unique_mods]
        fig.add_trace(
            go.Pie(labels=unique_mods, values=counts, name="Modalities"),
            row=1, col=2
        )
        
        # Panel 3: Embedding scatter (if provided)
        if embeddings is not None and len(embeddings) > 0:
            if embeddings.shape[1] > 2:
                coords = EmbeddingVisualizer.reduce_dimensions(embeddings, "pca", 2)
            else:
                coords = embeddings
            fig.add_trace(
                go.Scatter(
                    x=coords[:, 0], 
                    y=coords[:, 1], 
                    mode='markers',
                    text=labels,
                    name="Embeddings"
                ),
                row=2, col=1
            )
        
        # Panel 4: Similarity matrix (if embeddings provided)
        if embeddings is not None and len(embeddings) > 1:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / np.clip(norms, 1e-9, None)
            similarity = np.dot(normalized, normalized.T)
            fig.add_trace(
                go.Heatmap(z=similarity, colorscale='RdBu', name="Similarity"),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="BioFlow Dashboard")
        return fig
