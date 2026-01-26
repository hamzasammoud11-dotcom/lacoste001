"""
BioFlow Explorer - Streamlit Interface
=======================================

Interactive web interface for testing and exploring the BioFlow
multimodal biological intelligence system.

Run with: streamlit run bioflow/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import json
import os
import sys

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Page config
st.set_page_config(
    page_title="BioFlow Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .result-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
    }
    .modality-text { color: #3b82f6; }
    .modality-molecule { color: #10b981; }
    .modality-protein { color: #f59e0b; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_bioflow(use_mock: bool = True):
    """Initialize BioFlow components (cached)."""
    try:
        from bioflow.obm_wrapper import OBMWrapper
        from bioflow.qdrant_manager import QdrantManager
        from bioflow.pipeline import BioFlowPipeline, MinerAgent, ValidatorAgent
        
        obm = OBMWrapper(use_mock=use_mock)
        qdrant = QdrantManager(obm, qdrant_path=None)  # In-memory
        qdrant.create_collection("bioflow_demo", recreate=True)
        
        pipeline = BioFlowPipeline(obm, qdrant)
        pipeline.register_agent(MinerAgent(obm, qdrant, "bioflow_demo"))
        pipeline.register_agent(ValidatorAgent(obm, qdrant, "bioflow_demo"))
        
        return {
            "obm": obm,
            "qdrant": qdrant,
            "pipeline": pipeline,
            "ready": True
        }
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        return {"ready": False, "error": str(e)}


def render_sidebar():
    """Render the sidebar with controls."""
    st.sidebar.markdown("## 🧬 BioFlow Explorer")
    st.sidebar.markdown("---")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Mode",
        ["🔍 Search & Explore", "📥 Data Ingestion", "🧪 Cross-Modal Analysis", 
         "📊 Visualization", "🔬 Pipeline Demo", "📚 Documentation"]
    )
    
    st.sidebar.markdown("---")
    
    # Settings
    with st.sidebar.expander("⚙️ Settings"):
        use_mock = st.checkbox("Use Mock Mode (no GPU needed)", value=True)
        vector_dim = st.number_input("Vector Dimension", value=768, disabled=True)
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")
    
    return mode, use_mock


def render_search_page(components):
    """Render the search and explore page."""
    st.markdown('<p class="main-header">🔍 Search & Explore</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_area(
            "Enter your query",
            placeholder="e.g., 'KRAS inhibitor for cancer treatment' or a SMILES string like 'CCO'",
            height=100
        )
        
        query_modality = st.selectbox(
            "Query Modality",
            ["text", "smiles", "protein"],
            help="Select the type of your input"
        )
    
    with col2:
        target_modality = st.selectbox(
            "Search for",
            ["All", "text", "smiles", "protein"],
            help="Filter results by modality"
        )
        
        top_k = st.slider("Number of results", 1, 20, 5)
    
    if st.button("🔍 Search", type="primary"):
        if not query:
            st.warning("Please enter a query")
            return
        
        with st.spinner("Encoding and searching..."):
            obm = components["obm"]
            qdrant = components["qdrant"]
            
            # Encode query
            embedding = obm.encode(query, query_modality)
            
            # Display query embedding info
            with st.expander("📊 Query Embedding Details"):
                st.json({
                    "modality": embedding.modality.value,
                    "dimension": embedding.dimension,
                    "content_hash": embedding.content_hash,
                    "vector_sample": embedding.vector[:5].tolist()
                })
            
            # Search
            filter_mod = None if target_modality == "All" else target_modality
            results = qdrant.search(
                query=query,
                query_modality=query_modality,
                limit=top_k,
                filter_modality=filter_mod
            )
            
            if results:
                st.markdown("### 📋 Search Results")
                for i, r in enumerate(results):
                    with st.container():
                        col1, col2, col3 = st.columns([1, 4, 1])
                        with col1:
                            st.metric("Rank", i + 1)
                        with col2:
                            modality_class = f"modality-{r.modality}"
                            st.markdown(f"**<span class='{modality_class}'>[{r.modality.upper()}]</span>** {r.content[:100]}...", unsafe_allow_html=True)
                        with col3:
                            st.metric("Score", f"{r.score:.3f}")
                        st.divider()
            else:
                st.info("No results found. Try ingesting some data first!")


def render_ingestion_page(components):
    """Render the data ingestion page."""
    st.markdown('<p class="main-header">📥 Data Ingestion</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📝 Single Entry", "📄 Batch Upload", "🧪 Sample Data"])
    
    with tab1:
        st.markdown("### Add Single Entry")
        
        col1, col2 = st.columns(2)
        with col1:
            content = st.text_area("Content", placeholder="Enter text, SMILES, or protein sequence")
            modality = st.selectbox("Type", ["text", "smiles", "protein"])
        
        with col2:
            source = st.text_input("Source", placeholder="e.g., PubMed:12345")
            tags = st.text_input("Tags (comma-separated)", placeholder="e.g., cancer, kinase")
        
        if st.button("➕ Add Entry"):
            if content:
                qdrant = components["qdrant"]
                item = {
                    "content": content,
                    "modality": modality,
                    "source": source,
                    "tags": [t.strip() for t in tags.split(",") if t.strip()]
                }
                stats = qdrant.ingest([item])
                st.success(f"Added successfully! Stats: {stats}")
            else:
                st.warning("Please enter content")
    
    with tab2:
        st.markdown("### Batch Upload")
        
        uploaded_file = st.file_uploader("Upload JSON or CSV", type=["json", "csv"])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.json'):
                    data = json.load(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
                    data = df.to_dict('records')
                
                st.write(f"Found {len(data)} entries")
                st.dataframe(pd.DataFrame(data).head())
                
                if st.button("📤 Upload All"):
                    qdrant = components["qdrant"]
                    stats = qdrant.ingest(data)
                    st.success(f"Ingestion complete! {stats}")
            except Exception as e:
                st.error(f"Error parsing file: {e}")
    
    with tab3:
        st.markdown("### Load Sample Data")
        st.markdown("Load pre-defined sample data to test the system.")
        
        sample_data = [
            {"content": "Aspirin is used to reduce fever and relieve mild to moderate pain", "modality": "text", "source": "sample", "tags": ["pain", "fever"]},
            {"content": "CC(=O)OC1=CC=CC=C1C(=O)O", "modality": "smiles", "source": "ChEMBL", "tags": ["aspirin", "nsaid"]},
            {"content": "Ibuprofen is a nonsteroidal anti-inflammatory drug used for treating pain", "modality": "text", "source": "sample", "tags": ["pain", "nsaid"]},
            {"content": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "modality": "smiles", "source": "ChEMBL", "tags": ["ibuprofen", "nsaid"]},
            {"content": "KRAS mutations are found in many cancers and are difficult to target", "modality": "text", "source": "PubMed", "tags": ["cancer", "KRAS"]},
            {"content": "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM", "modality": "protein", "source": "UniProt:P01116", "tags": ["KRAS", "GTPase"]},
            {"content": "Sotorasib is a first-in-class KRAS G12C inhibitor", "modality": "text", "source": "PubMed", "tags": ["KRAS", "inhibitor", "cancer"]},
            {"content": "C[C@@H]1CC(=O)N(C2=C1C=CC(=C2)NC(=O)C3=CC=C(C=C3)N4CCN(CC4)C)C5=NC=CC(=N5)C6CCCCC6", "modality": "smiles", "source": "ChEMBL", "tags": ["sotorasib", "KRAS", "inhibitor"]},
        ]
        
        if st.button("🧪 Load Sample Data"):
            qdrant = components["qdrant"]
            stats = qdrant.ingest(sample_data)
            st.success(f"Loaded {len(sample_data)} sample entries! {stats}")
            st.balloons()


def render_crossmodal_page(components):
    """Render cross-modal analysis page."""
    st.markdown('<p class="main-header">🧪 Cross-Modal Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore how different modalities relate to each other in the shared embedding space.
    This is the core capability of BioFlow - connecting text, molecules, and proteins.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Query")
        query = st.text_area("Enter query", height=100)
        query_mod = st.selectbox("Query type", ["text", "smiles", "protein"], key="q_mod")
    
    with col2:
        st.markdown("### Targets")
        targets = st.text_area("Enter targets (one per line)", height=100)
        target_mod = st.selectbox("Target type", ["text", "smiles", "protein"], key="t_mod")
    
    if st.button("🔄 Compute Cross-Modal Similarity"):
        if query and targets:
            obm = components["obm"]
            target_list = [t.strip() for t in targets.strip().split("\n") if t.strip()]
            
            results = obm.cross_modal_similarity(
                query=query,
                query_modality=query_mod,
                targets=target_list,
                target_modality=target_mod
            )
            
            st.markdown("### Results (sorted by similarity)")
            
            df = pd.DataFrame(results, columns=["Content", "Similarity"])
            df["Rank"] = range(1, len(df) + 1)
            df = df[["Rank", "Content", "Similarity"]]
            
            st.dataframe(df, use_container_width=True)
            
            # Visualize
            import plotly.express as px
            fig = px.bar(df, x="Content", y="Similarity", title="Cross-Modal Similarities")
            st.plotly_chart(fig, use_container_width=True)


def render_visualization_page(components):
    """Render visualization page."""
    st.markdown('<p class="main-header">📊 Visualization</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🌐 Embedding Space", "📈 Similarity Matrix", "🧬 Molecules"])
    
    with tab1:
        st.markdown("### Embedding Space Visualization")
        
        # Get all points from collection
        qdrant = components["qdrant"]
        info = qdrant.get_collection_info()
        
        if info.get("points_count", 0) == 0:
            st.warning("No data in collection. Go to Data Ingestion to add some!")
            return
        
        st.metric("Points in collection", info.get("points_count", 0))
        
        if st.button("🎨 Generate Embedding Plot"):
            # This would require fetching all vectors - simplified for demo
            st.info("Embedding visualization requires fetching all vectors. In production, use sampling.")
            
            # Demo with random data
            n_points = min(info.get("points_count", 20), 50)
            fake_embeddings = np.random.randn(n_points, 2)
            
            import plotly.express as px
            fig = px.scatter(
                x=fake_embeddings[:, 0],
                y=fake_embeddings[:, 1],
                title="Embedding Space (Demo - PCA projection)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Compute Similarity Matrix")
        
        items = st.text_area("Enter items (one per line)", height=150)
        modality = st.selectbox("Modality", ["text", "smiles", "protein"], key="sim_mod")
        
        if st.button("🔢 Compute Matrix"):
            if items:
                obm = components["obm"]
                item_list = [i.strip() for i in items.strip().split("\n") if i.strip()]
                
                if modality == "text":
                    embeddings = obm.encode_text(item_list)
                elif modality == "smiles":
                    embeddings = obm.encode_smiles(item_list)
                else:
                    embeddings = obm.encode_protein(item_list)
                
                vectors = np.array([e.vector for e in embeddings])
                
                # Compute similarity
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                normalized = vectors / np.clip(norms, 1e-9, None)
                similarity = np.dot(normalized, normalized.T)
                
                import plotly.figure_factory as ff
                labels = [i[:20] for i in item_list]
                fig = ff.create_annotated_heatmap(
                    similarity,
                    x=labels,
                    y=labels,
                    colorscale='RdBu'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Molecule Visualization")
        
        smiles = st.text_input("Enter SMILES", placeholder="CC(=O)OC1=CC=CC=C1C(=O)O")
        
        if smiles:
            try:
                from rdkit import Chem
                from rdkit.Chem import Draw
                
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    img = Draw.MolToImage(mol, size=(400, 300))
                    st.image(img, caption=f"Molecule: {smiles}")
                else:
                    st.error("Invalid SMILES")
            except ImportError:
                st.warning("RDKit not installed. Install with: pip install rdkit")


def render_pipeline_page(components):
    """Render pipeline demo page."""
    st.markdown('<p class="main-header">🔬 Pipeline Demo</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Run a complete discovery workflow that:
    1. Searches for related literature
    2. Finds similar molecules
    3. Validates candidates
    4. Analyzes result diversity
    """)
    
    query = st.text_input("Enter discovery query", placeholder="e.g., KRAS inhibitor for lung cancer")
    
    col1, col2 = st.columns(2)
    with col1:
        query_mod = st.selectbox("Query modality", ["text", "smiles", "protein"])
    with col2:
        target_mod = st.selectbox("Target modality", ["smiles", "text", "protein"])
    
    if st.button("🚀 Run Discovery Pipeline", type="primary"):
        if query:
            pipeline = components["pipeline"]
            
            with st.spinner("Running pipeline..."):
                results = pipeline.run_discovery_workflow(
                    query=query,
                    query_modality=query_mod,
                    target_modality=target_mod
                )
            
            st.markdown("## 📊 Pipeline Results")
            
            # Literature
            with st.expander("📚 Related Literature", expanded=True):
                lit = results.get("stages", {}).get("literature", [])
                if lit:
                    for item in lit:
                        st.markdown(f"- **Score: {item['score']:.3f}** - {item['content'][:100]}...")
                else:
                    st.info("No literature found")
            
            # Molecules
            with st.expander("🧪 Similar Molecules", expanded=True):
                mols = results.get("stages", {}).get("molecules", [])
                if mols:
                    df = pd.DataFrame(mols)
                    st.dataframe(df)
                else:
                    st.info("No molecules found")
            
            # Validation
            with st.expander("✅ Validation Results"):
                val = results.get("stages", {}).get("validation", [])
                if val:
                    st.json(val)
                else:
                    st.info("No validation performed")
            
            # Diversity
            with st.expander("📈 Diversity Analysis"):
                div = results.get("stages", {}).get("diversity", {})
                if div:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Similarity", f"{div.get('mean_similarity', 0):.3f}")
                    col2.metric("Diversity Score", f"{div.get('diversity_score', 0):.3f}")
                    col3.metric("Modalities", len(div.get('modality_distribution', {})))
                    st.json(div)


def render_docs_page():
    """Render documentation page."""
    st.markdown('<p class="main-header">📚 Documentation</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## BioFlow + OpenBioMed Integration
    
    ### 🎯 Overview
    
    BioFlow is a multimodal biological intelligence framework that leverages OpenBioMed (OBM)
    for encoding biological data and Qdrant for vector storage and retrieval.
    
    ### 🧩 Components
    
    | Component | Description |
    |-----------|-------------|
    | **OBMWrapper** | Encodes text, molecules (SMILES), and proteins into a shared vector space |
    | **QdrantManager** | Manages vector storage, indexing, and similarity search |
    | **BioFlowPipeline** | Orchestrates agents in discovery workflows |
    | **Visualizer** | Creates plots for embeddings, similarities, and molecules |
    
    ### 🔌 API Examples
    
    ```python
    from bioflow import OBMWrapper, QdrantManager, BioFlowPipeline
    
    # Initialize
    obm = OBMWrapper(device="cuda")
    qdrant = QdrantManager(obm, qdrant_path="./data/qdrant")
    
    # Encode different modalities
    text_vec = obm.encode_text("KRAS inhibitor for cancer")
    mol_vec = obm.encode_smiles("CCO")
    prot_vec = obm.encode_protein("MTEYKLVVV...")
    
    # Cross-modal search
    results = qdrant.cross_modal_search(
        query="anti-inflammatory drug",
        query_modality="text",
        target_modality="smiles",
        limit=10
    )
    ```
    
    ### 🌟 Key Features
    
    1. **Unified Embedding Space**: All modalities map to the same vector dimension
    2. **Cross-Modal Search**: Find molecules from text queries and vice versa
    3. **Pipeline Orchestration**: Chain agents for complex discovery workflows
    4. **Mock Mode**: Test without GPU using deterministic random embeddings
    
    ### 📁 File Structure
    
    ```
    bioflow/
    ├── __init__.py          # Package exports
    ├── obm_wrapper.py       # OBM encoding interface
    ├── qdrant_manager.py    # Qdrant operations
    ├── pipeline.py          # Workflow orchestration
    ├── visualizer.py        # Visualization utilities
    └── app.py               # Streamlit interface
    ```
    """)


def main():
    """Main application entry point."""
    mode, use_mock = render_sidebar()
    
    # Initialize components
    components = init_bioflow(use_mock=use_mock)
    
    if not components.get("ready"):
        st.error("System not ready. Check configuration.")
        return
    
    # Display collection stats in sidebar
    info = components["qdrant"].get_collection_info()
    st.sidebar.metric("📊 Vectors", info.get("points_count", 0))
    st.sidebar.metric("📐 Dimension", info.get("vector_size", 768))
    
    # Route to appropriate page
    if "Search" in mode:
        render_search_page(components)
    elif "Ingestion" in mode:
        render_ingestion_page(components)
    elif "Cross-Modal" in mode:
        render_crossmodal_page(components)
    elif "Visualization" in mode:
        render_visualization_page(components)
    elif "Pipeline" in mode:
        render_pipeline_page(components)
    elif "Documentation" in mode:
        render_docs_page()


if __name__ == "__main__":
    main()
