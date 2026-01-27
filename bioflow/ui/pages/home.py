"""
BioFlow - Home Page
====================
Clean dashboard with key metrics and quick actions.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from bioflow.ui.components import (
    section_header, divider, spacer,
    metric_card, pipeline_progress, bar_chart
)
from bioflow.ui.config import COLORS


def render():
    """Render home page."""
    
    # Hero Section (Tailark-inspired)
    hero_col, hero_side = st.columns([3, 1.4])
    
    with hero_col:
        st.markdown(f"""
        <div class="hero">
            <div class="hero-badge">New • BioFlow 2.0</div>
            <div class="hero-title">AI-Powered Drug Discovery</div>
            <div class="hero-subtitle">
                Run discovery pipelines, predict binding, and surface evidence in one streamlined workspace.
            </div>
            <div class="hero-actions">
                <span class="badge badge-primary">Model-aware search</span>
                <span class="badge badge-success">Evidence-linked</span>
                <span class="badge badge-warning">Fast iteration</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        spacer("0.75rem")
        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("Start Discovery", type="primary", use_container_width=True):
                st.session_state.current_page = "discovery"
                st.rerun()
        with btn2:
            if st.button("Explore Data", use_container_width=True):
                st.session_state.current_page = "explorer"
                st.rerun()
    
    with hero_side:
        st.markdown(f"""
        <div class="hero-card">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: {COLORS.text_muted};">
                Today
            </div>
            <div style="font-size: 1.75rem; font-weight: 700; color: {COLORS.text_primary}; margin-top: 0.5rem;">
                156 Discoveries
            </div>
            <div style="font-size: 0.875rem; color: {COLORS.text_muted}; margin-top: 0.5rem;">
                +12% vs last week
            </div>
            <div class="divider" style="margin: 1rem 0;"></div>
            <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                <span class="badge badge-primary">Discovery</span>
                <span class="badge badge-success">Prediction</span>
                <span class="badge badge-warning">Evidence</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics Row
    cols = st.columns(4)
    
    with cols[0]:
        metric_card("12.5M", "Molecules", "🧪", "+2.3%", "up", COLORS.primary)
    with cols[1]:
        metric_card("847K", "Proteins", "🧬", "+1.8%", "up", COLORS.cyan)
    with cols[2]:
        metric_card("1.2M", "Papers", "📚", "+5.2%", "up", COLORS.emerald)
    with cols[3]:
        metric_card("156", "Discoveries", "✨", "+12%", "up", COLORS.amber)
    
    spacer("2rem")
    
    # Quick Actions
    section_header("Quick Actions", "⚡")
    
    action_cols = st.columns(4)
    
    with action_cols[0]:
        st.markdown(f"""
        <div class="quick-action">
            <span class="quick-action-icon">🔍</span>
            <div class="quick-action-title">New Discovery</div>
            <div class="quick-action-desc">Start a pipeline</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start", key="qa_discovery", use_container_width=True):
            st.session_state.current_page = "discovery"
            st.rerun()
    
    with action_cols[1]:
        st.markdown(f"""
        <div class="quick-action">
            <span class="quick-action-icon">📊</span>
            <div class="quick-action-title">Explore Data</div>
            <div class="quick-action-desc">Visualize embeddings</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore", key="qa_explorer", use_container_width=True):
            st.session_state.current_page = "explorer"
            st.rerun()
    
    with action_cols[2]:
        st.markdown(f"""
        <div class="quick-action">
            <span class="quick-action-icon">📁</span>
            <div class="quick-action-title">Upload Data</div>
            <div class="quick-action-desc">Add molecules</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Upload", key="qa_data", use_container_width=True):
            st.session_state.current_page = "data"
            st.rerun()
    
    with action_cols[3]:
        st.markdown(f"""
        <div class="quick-action">
            <span class="quick-action-icon">⚙️</span>
            <div class="quick-action-title">Settings</div>
            <div class="quick-action-desc">Configure models</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Configure", key="qa_settings", use_container_width=True):
            st.session_state.current_page = "settings"
            st.rerun()
    
    spacer("2rem")
    divider()
    spacer("2rem")
    
    # Two Column Layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        section_header("Recent Discoveries", "🎯")
        
        # Sample results
        results = [
            {"name": "Aspirin analog", "score": 0.94, "mw": "180.16"},
            {"name": "Novel kinase inhibitor", "score": 0.87, "mw": "331.39"},
            {"name": "EGFR binder candidate", "score": 0.72, "mw": "311.38"},
        ]
        
        for r in results:
            score_color = COLORS.emerald if r["score"] >= 0.8 else (COLORS.amber if r["score"] >= 0.5 else COLORS.rose)
            st.markdown(f"""
            <div class="result">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 600; color: {COLORS.text_primary};">{r["name"]}</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: {score_color};">{r["score"]:.0%}</div>
                </div>
                <div style="font-size: 0.8125rem; color: {COLORS.text_muted}; margin-top: 0.5rem;">
                    MW: {r["mw"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            spacer("0.75rem")
    
    with col2:
        section_header("Pipeline Activity", "📈")
        
        bar_chart(
            {"Mon": 23, "Tue": 31, "Wed": 28, "Thu": 45, "Fri": 38, "Sat": 12, "Sun": 8},
            title="",
            height=250
        )
        
        spacer("1rem")
        section_header("Active Pipeline", "🔄")
        
        pipeline_progress([
            {"name": "Encode", "status": "done"},
            {"name": "Search", "status": "active"},
            {"name": "Predict", "status": "pending"},
            {"name": "Verify", "status": "pending"},
        ])
    
    spacer("2rem")
    
    # Tip
    st.markdown(f"""
    <div style="
        background: {COLORS.bg_surface};
        border: 1px solid {COLORS.border};
        border-radius: 12px;
        padding: 1.25rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    ">
        <span style="font-size: 1.5rem;">💡</span>
        <div>
            <div style="font-size: 0.9375rem; color: {COLORS.text_primary}; font-weight: 500;">Pro Tip</div>
            <div style="font-size: 0.8125rem; color: {COLORS.text_muted};">
                Use natural language like "Find molecules similar to aspirin that can cross the blood-brain barrier"
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
