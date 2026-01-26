"""
BioFlow - Discovery Page
========================
Drug discovery pipeline interface.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from bioflow.ui.components import (
    page_header, section_header, divider, spacer,
    pipeline_progress, bar_chart, empty_state, loading_state
)
from bioflow.ui.config import COLORS


def render():
    """Render discovery page."""
    
    page_header("Drug Discovery", "Search for drug candidates with AI-powered analysis", "🔬")
    
    # Query Input Section
    st.markdown(f"""
    <div class="card" style="margin-bottom: 1.5rem;">
        <div style="font-size: 0.875rem; font-weight: 600; color: {COLORS.text_primary}; margin-bottom: 0.75rem;">
            Search Query
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Query",
            placeholder="Enter a natural language query, SMILES string, or FASTA sequence...",
            height=100,
            label_visibility="collapsed"
        )
    
    with col2:
        st.selectbox("Search Type", ["Similarity", "Binding Affinity", "Properties"], label_visibility="collapsed")
        st.selectbox("Database", ["All", "DrugBank", "ChEMBL", "ZINC"], label_visibility="collapsed")
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
    
    spacer("1.5rem")
    
    # Pipeline Progress
    section_header("Pipeline Status", "🔄")
    
    if "discovery_step" not in st.session_state:
        st.session_state.discovery_step = 0
    
    steps = [
        {"name": "Input", "status": "done" if st.session_state.discovery_step > 0 else "active"},
        {"name": "Encode", "status": "done" if st.session_state.discovery_step > 1 else ("active" if st.session_state.discovery_step == 1 else "pending")},
        {"name": "Search", "status": "done" if st.session_state.discovery_step > 2 else ("active" if st.session_state.discovery_step == 2 else "pending")},
        {"name": "Predict", "status": "done" if st.session_state.discovery_step > 3 else ("active" if st.session_state.discovery_step == 3 else "pending")},
        {"name": "Results", "status": "active" if st.session_state.discovery_step == 4 else "pending"},
    ]
    
    pipeline_progress(steps)
    
    spacer("2rem")
    divider()
    spacer("2rem")
    
    # Results Section
    section_header("Results", "🎯")
    
    if search_clicked and query:
        st.session_state.discovery_step = 4
        st.session_state.discovery_query = query
    
    if st.session_state.discovery_step >= 4:
        # Show results
        tabs = st.tabs(["Top Candidates", "Property Analysis", "Evidence"])
        
        with tabs[0]:
            # Results list
            results = [
                {"name": "Candidate A", "score": 0.95, "mw": "342.4", "logp": "2.1", "hbd": "2"},
                {"name": "Candidate B", "score": 0.89, "mw": "298.3", "logp": "1.8", "hbd": "3"},
                {"name": "Candidate C", "score": 0.82, "mw": "415.5", "logp": "3.2", "hbd": "1"},
                {"name": "Candidate D", "score": 0.76, "mw": "267.3", "logp": "1.5", "hbd": "2"},
                {"name": "Candidate E", "score": 0.71, "mw": "389.4", "logp": "2.8", "hbd": "2"},
            ]
            
            for r in results:
                score_color = COLORS.emerald if r["score"] >= 0.8 else (COLORS.amber if r["score"] >= 0.5 else COLORS.rose)
                st.markdown(f"""
                <div class="result">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <div style="font-weight: 600; color: {COLORS.text_primary};">{r["name"]}</div>
                            <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
                                <span style="font-size: 0.8125rem; color: {COLORS.text_muted};">MW: {r["mw"]}</span>
                                <span style="font-size: 0.8125rem; color: {COLORS.text_muted};">LogP: {r["logp"]}</span>
                                <span style="font-size: 0.8125rem; color: {COLORS.text_muted};">HBD: {r["hbd"]}</span>
                            </div>
                        </div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: {score_color};">{r["score"]:.0%}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                spacer("0.75rem")
        
        with tabs[1]:
            # Property distribution
            col1, col2 = st.columns(2)
            
            with col1:
                bar_chart(
                    {"<200": 5, "200-300": 12, "300-400": 8, "400-500": 3, ">500": 2},
                    title="Molecular Weight Distribution",
                    height=250
                )
            
            with col2:
                bar_chart(
                    {"<1": 4, "1-2": 10, "2-3": 8, "3-4": 5, ">4": 3},
                    title="LogP Distribution",
                    height=250
                )
        
        with tabs[2]:
            # Evidence
            st.markdown(f"""
            <div class="card">
                <div style="font-weight: 600; color: {COLORS.text_primary}; margin-bottom: 1rem;">
                    Related Literature
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            papers = [
                {"title": "Novel therapeutic targets for cancer treatment", "year": "2024", "journal": "Nature Medicine"},
                {"title": "Molecular docking studies of kinase inhibitors", "year": "2023", "journal": "J. Med. Chem."},
                {"title": "AI-driven drug discovery approaches", "year": "2024", "journal": "Drug Discovery Today"},
            ]
            
            for p in papers:
                st.markdown(f"""
                <div style="
                    padding: 1rem;
                    border: 1px solid {COLORS.border};
                    border-radius: 8px;
                    margin-bottom: 0.75rem;
                ">
                    <div style="font-weight: 500; color: {COLORS.text_primary};">{p["title"]}</div>
                    <div style="font-size: 0.8125rem; color: {COLORS.text_muted}; margin-top: 0.25rem;">
                        {p["journal"]} • {p["year"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        empty_state(
            "🔍",
            "No Results Yet",
            "Enter a query and click Search to find drug candidates"
        )
