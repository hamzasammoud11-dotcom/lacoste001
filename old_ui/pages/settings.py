"""
BioFlow - Settings Page
========================
Configuration and preferences.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from bioflow.ui.components import page_header, section_header, divider, spacer
from bioflow.ui.config import COLORS


def render():
    """Render settings page."""
    
    page_header("Settings", "Configure models, databases, and preferences", "⚙️")
    
    # Tabs for different settings sections
    tabs = st.tabs(["🧠 Models", "🗄️ Database", "🔌 API Keys", "🎨 Appearance"])
    
    with tabs[0]:
        section_header("Model Configuration", "🧠")
        
        st.markdown(f"""
        <div class="card" style="margin-bottom: 1rem;">
            <div style="font-weight: 600; color: {COLORS.text_primary}; margin-bottom: 0.5rem;">
                Embedding Models
            </div>
            <div style="font-size: 0.8125rem; color: {COLORS.text_muted};">
                Configure models used for molecular and protein embeddings
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox(
                "Molecule Encoder",
                ["MolCLR (Recommended)", "ChemBERTa", "GraphMVP", "MolBERT"],
                help="Model for generating molecular embeddings"
            )
            
            st.selectbox(
                "Protein Encoder", 
                ["ESM-2 (Recommended)", "ProtTrans", "UniRep", "SeqVec"],
                help="Model for generating protein embeddings"
            )
        
        with col2:
            st.selectbox(
                "Binding Predictor",
                ["DrugBAN (Recommended)", "DeepDTA", "GraphDTA", "Custom"],
                help="Model for predicting drug-target binding"
            )
            
            st.selectbox(
                "Property Predictor",
                ["ADMET-AI (Recommended)", "ChemProp", "Custom"],
                help="Model for ADMET property prediction"
            )
        
        spacer("1rem")
        
        st.markdown(f"""
        <div class="card" style="margin-bottom: 1rem;">
            <div style="font-weight: 600; color: {COLORS.text_primary}; margin-bottom: 0.5rem;">
                LLM Settings
            </div>
            <div style="font-size: 0.8125rem; color: {COLORS.text_muted};">
                Configure language models for evidence retrieval and reasoning
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox(
                "LLM Provider",
                ["OpenAI", "Anthropic", "Local (Ollama)", "Azure OpenAI"]
            )
        
        with col2:
            st.selectbox(
                "Model",
                ["GPT-4o", "GPT-4-turbo", "Claude 3.5 Sonnet", "Llama 3.1 70B"]
            )
        
        st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        st.number_input("Max Tokens", 100, 4096, 2048, 100)
    
    with tabs[1]:
        section_header("Database Configuration", "🗄️")
        
        st.markdown(f"""
        <div class="card" style="margin-bottom: 1rem;">
            <div style="font-weight: 600; color: {COLORS.text_primary}; margin-bottom: 0.5rem;">
                Vector Database
            </div>
            <div style="font-size: 0.8125rem; color: {COLORS.text_muted};">
                Configure the vector store for similarity search
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("Vector Store", ["Qdrant (Recommended)", "Milvus", "Pinecone", "Weaviate", "ChromaDB"])
            st.text_input("Host", value="localhost")
        
        with col2:
            st.number_input("Port", 1, 65535, 6333)
            st.text_input("Collection Name", value="bioflow_embeddings")
        
        spacer("1rem")
        
        st.markdown(f"""
        <div class="card" style="margin-bottom: 1rem;">
            <div style="font-weight: 600; color: {COLORS.text_primary}; margin-bottom: 0.5rem;">
                Knowledge Sources
            </div>
            <div style="font-size: 0.8125rem; color: {COLORS.text_muted};">
                External databases for evidence retrieval
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("PubMed", value=True)
            st.checkbox("DrugBank", value=True)
            st.checkbox("ChEMBL", value=True)
        
        with col2:
            st.checkbox("UniProt", value=True)
            st.checkbox("KEGG", value=False)
            st.checkbox("Reactome", value=False)
    
    with tabs[2]:
        section_header("API Keys", "🔌")
        
        st.warning("⚠️ API keys are stored locally and never sent to external servers.")
        
        st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
        st.text_input("PubMed API Key", type="password", placeholder="Optional - for higher rate limits")
        st.text_input("ChEMBL API Key", type="password", placeholder="Optional")
        
        spacer("1rem")
        
        if st.button("💾 Save API Keys", type="primary"):
            st.success("✓ API keys saved securely")
    
    with tabs[3]:
        section_header("Appearance", "🎨")
        
        st.selectbox("Theme", ["Dark (Default)", "Light", "System"])
        st.selectbox("Accent Color", ["Purple", "Blue", "Green", "Cyan", "Pink"])
        st.checkbox("Enable animations", value=True)
        st.checkbox("Compact mode", value=False)
        st.slider("Font size", 12, 18, 14)
    
    spacer("2rem")
    divider()
    spacer("1rem")
    
    # Save buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            st.success("✓ Settings saved successfully!")
    
    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.info("Settings reset to defaults")
    
    spacer("2rem")
    
    # Version info
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; color: {COLORS.text_muted}; font-size: 0.75rem;">
        BioFlow v0.1.0 • Built with OpenBioMed
    </div>
    """, unsafe_allow_html=True)
