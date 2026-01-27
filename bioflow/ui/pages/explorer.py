"""
BioFlow - Explorer Page
=======================
Data exploration and visualization.
"""

import streamlit as st
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from bioflow.ui.components import (
    page_header, section_header, divider, spacer,
    scatter_chart, heatmap, metric_card, empty_state
)
from bioflow.ui.config import COLORS


def render():
    """Render explorer page."""
    
    page_header("Data Explorer", "Visualize molecular embeddings and relationships", "🧬")
    
    # Controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dataset = st.selectbox("Dataset", ["DrugBank", "ChEMBL", "ZINC", "Custom"])
    with col2:
        viz_type = st.selectbox("Visualization", ["UMAP", "t-SNE", "PCA"])
    with col3:
        color_by = st.selectbox("Color by", ["Activity", "MW", "LogP", "Cluster"])
    with col4:
        st.write("")  # Spacing
        st.write("")
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    spacer("1.5rem")
    
    # Main visualization area
    col_viz, col_details = st.columns([2, 1])
    
    with col_viz:
        section_header("Embedding Space", "🗺️")
        
        # Generate sample data
        np.random.seed(42)
        n_points = 200
        
        # Create clusters
        cluster1_x = np.random.normal(2, 0.8, n_points // 4)
        cluster1_y = np.random.normal(3, 0.8, n_points // 4)
        
        cluster2_x = np.random.normal(-2, 1, n_points // 4)
        cluster2_y = np.random.normal(-1, 1, n_points // 4)
        
        cluster3_x = np.random.normal(4, 0.6, n_points // 4)
        cluster3_y = np.random.normal(-2, 0.6, n_points // 4)
        
        cluster4_x = np.random.normal(-1, 0.9, n_points // 4)
        cluster4_y = np.random.normal(4, 0.9, n_points // 4)
        
        x = list(cluster1_x) + list(cluster2_x) + list(cluster3_x) + list(cluster4_x)
        y = list(cluster1_y) + list(cluster2_y) + list(cluster3_y) + list(cluster4_y)
        labels = [f"Mol_{i}" for i in range(n_points)]
        
        scatter_chart(x, y, labels, title=f"{viz_type} Projection - {dataset}", height=450)
    
    with col_details:
        section_header("Statistics", "📊")
        
        metric_card("12,450", "Total Molecules", "🧪", color=COLORS.primary)
        spacer("0.75rem")
        metric_card("4", "Clusters Found", "🎯", color=COLORS.cyan)
        spacer("0.75rem")
        metric_card("0.89", "Silhouette Score", "📈", color=COLORS.emerald)
        spacer("0.75rem")
        metric_card("85%", "Coverage", "✓", color=COLORS.amber)
    
    spacer("2rem")
    divider()
    spacer("2rem")
    
    # Similarity Heatmap
    section_header("Similarity Matrix", "🔥")
    
    # Sample similarity matrix
    np.random.seed(123)
    labels_short = ["Cluster A", "Cluster B", "Cluster C", "Cluster D", "Cluster E"]
    similarity = np.random.uniform(0.3, 1.0, (5, 5))
    similarity = (similarity + similarity.T) / 2  # Make symmetric
    np.fill_diagonal(similarity, 1.0)
    
    heatmap(
        similarity.tolist(),
        labels_short,
        labels_short,
        title="Inter-cluster Similarity",
        height=350
    )
    
    spacer("2rem")
    
    # Export options
    st.markdown(f"""
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-weight: 600; color: {COLORS.text_primary};">Export Data</div>
                <div style="font-size: 0.8125rem; color: {COLORS.text_muted}; margin-top: 0.25rem;">
                    Download embeddings, clusters, or full dataset
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    exp_cols = st.columns(3)
    with exp_cols[0]:
        st.button("📥 Embeddings (CSV)", use_container_width=True)
    with exp_cols[1]:
        st.button("📥 Clusters (JSON)", use_container_width=True)
    with exp_cols[2]:
        st.button("📥 Full Dataset", use_container_width=True)
