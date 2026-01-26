"""
BioFlow - AI-Powered Drug Discovery Platform
==============================================
Main application entry point.
"""

import streamlit as st
import sys
import os

# Setup path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bioflow.ui.config import get_css
from bioflow.ui.components import side_nav
from bioflow.ui.pages import home, discovery, explorer, data, settings


def main():
    """Main application."""
    
    # Page config
    st.set_page_config(
        page_title="BioFlow",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    # Inject custom CSS
    st.markdown(get_css(), unsafe_allow_html=True)
    
    # Initialize session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    
    # Layout with left navigation
    nav_col, content_col = st.columns([1, 3.6], gap="large")
    
    with nav_col:
        selected = side_nav(active_page=st.session_state.current_page)
    
    if selected != st.session_state.current_page:
        st.session_state.current_page = selected
        st.rerun()
    
    with content_col:
        page_map = {
            "home": home.render,
            "discovery": discovery.render,
            "explorer": explorer.render,
            "data": data.render,
            "settings": settings.render,
        }
        
        render_fn = page_map.get(st.session_state.current_page, home.render)
        render_fn()


if __name__ == "__main__":
    main()
