"""
BioFlow - Data Page
===================
Data management and upload.
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from bioflow.ui.components import (
    page_header, section_header, divider, spacer,
    metric_card, data_table, empty_state
)
from bioflow.ui.config import COLORS


def render():
    """Render data page."""
    
    page_header("Data Management", "Upload, manage, and organize your datasets", "📊")
    
    # Stats Row
    cols = st.columns(4)
    
    with cols[0]:
        metric_card("5", "Datasets", "📁", color=COLORS.primary)
    with cols[1]:
        metric_card("24.5K", "Molecules", "🧪", color=COLORS.cyan)
    with cols[2]:
        metric_card("1.2K", "Proteins", "🧬", color=COLORS.emerald)
    with cols[3]:
        metric_card("156 MB", "Storage Used", "💾", color=COLORS.amber)
    
    spacer("2rem")
    
    # Tabs
    tabs = st.tabs(["📁 Datasets", "📤 Upload", "🔧 Processing"])
    
    with tabs[0]:
        section_header("Your Datasets", "📁")
        
        # Dataset list
        datasets = [
            {"name": "DrugBank Compounds", "type": "Molecules", "count": "12,450", "size": "45.2 MB", "updated": "2024-01-15"},
            {"name": "ChEMBL Kinase Inhibitors", "type": "Molecules", "count": "8,234", "size": "32.1 MB", "updated": "2024-01-10"},
            {"name": "Custom Protein Targets", "type": "Proteins", "count": "1,245", "size": "78.5 MB", "updated": "2024-01-08"},
        ]
        
        for ds in datasets:
            st.markdown(f"""
            <div class="card" style="margin-bottom: 0.75rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 600; color: {COLORS.text_primary};">{ds["name"]}</div>
                        <div style="display: flex; gap: 1.5rem; margin-top: 0.5rem;">
                            <span style="font-size: 0.8125rem; color: {COLORS.text_muted};">
                                <span style="color: {COLORS.primary};">●</span> {ds["type"]}
                            </span>
                            <span style="font-size: 0.8125rem; color: {COLORS.text_muted};">{ds["count"]} items</span>
                            <span style="font-size: 0.8125rem; color: {COLORS.text_muted};">{ds["size"]}</span>
                            <span style="font-size: 0.8125rem; color: {COLORS.text_muted};">Updated: {ds["updated"]}</span>
                        </div>
                    </div>
                    <div style="display: flex; gap: 0.5rem;">
                        <span class="badge badge-primary">{ds["type"]}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons
            btn_cols = st.columns([1, 1, 1, 4])
            with btn_cols[0]:
                st.button("View", key=f"view_{ds['name']}", use_container_width=True)
            with btn_cols[1]:
                st.button("Export", key=f"export_{ds['name']}", use_container_width=True)
            with btn_cols[2]:
                st.button("Delete", key=f"delete_{ds['name']}", use_container_width=True)
            
            spacer("0.5rem")
    
    with tabs[1]:
        section_header("Upload New Data", "📤")
        
        # Upload area
        st.markdown(f"""
        <div style="
            border: 2px dashed {COLORS.border};
            border-radius: 16px;
            padding: 3rem;
            text-align: center;
            background: {COLORS.bg_surface};
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
            <div style="font-size: 1.125rem; font-weight: 600; color: {COLORS.text_primary};">
                Drag & drop files here
            </div>
            <div style="font-size: 0.875rem; color: {COLORS.text_muted}; margin-top: 0.5rem;">
                or click to browse
            </div>
            <div style="font-size: 0.75rem; color: {COLORS.text_muted}; margin-top: 1rem;">
                Supports: CSV, SDF, FASTA, PDB, JSON
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload file",
            type=["csv", "sdf", "fasta", "pdb", "json"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            st.success(f"✓ File uploaded: {uploaded_file.name}")
            
            col1, col2 = st.columns(2)
            with col1:
                dataset_name = st.text_input("Dataset Name", value=uploaded_file.name.split('.')[0])
            with col2:
                data_type = st.selectbox("Data Type", ["Molecules", "Proteins", "Text"])
            
            if st.button("Process & Import", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    import time
                    time.sleep(2)
                st.success("✓ Dataset imported successfully!")
    
    with tabs[2]:
        section_header("Data Processing", "🔧")
        
        st.markdown(f"""
        <div class="card">
            <div style="font-weight: 600; color: {COLORS.text_primary}; margin-bottom: 0.75rem;">
                Available Operations
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        operations = [
            {"icon": "🧹", "name": "Clean & Validate", "desc": "Remove duplicates, fix invalid structures"},
            {"icon": "🔢", "name": "Compute Descriptors", "desc": "Calculate molecular properties and fingerprints"},
            {"icon": "🧠", "name": "Generate Embeddings", "desc": "Create vector representations using AI models"},
            {"icon": "🔗", "name": "Merge Datasets", "desc": "Combine multiple datasets with deduplication"},
        ]
        
        for op in operations:
            st.markdown(f"""
            <div class="quick-action" style="margin-bottom: 0.75rem; text-align: left;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="font-size: 1.5rem;">{op["icon"]}</span>
                    <div>
                        <div style="font-weight: 600; color: {COLORS.text_primary};">{op["name"]}</div>
                        <div style="font-size: 0.8125rem; color: {COLORS.text_muted};">{op["desc"]}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.button(f"Run {op['name']}", key=f"op_{op['name']}", use_container_width=True)
