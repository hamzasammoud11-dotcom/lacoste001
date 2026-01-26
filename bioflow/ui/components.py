"""
BioFlow UI - Components Library
================================
Reusable, modern UI components for Streamlit.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Callable
import plotly.express as px
import plotly.graph_objects as go

# Import colors
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bioflow.ui.config import COLORS


# === Navigation ===

def side_nav(active_page: str = "home") -> str:
    """Left vertical navigation list. Returns the selected page key."""

    nav_items = [
        ("home", "🏠", "Home"),
        ("discovery", "🔬", "Discovery"),
        ("explorer", "🧬", "Explorer"),
        ("data", "📊", "Data"),
        ("settings", "⚙️", "Settings"),
    ]

    st.markdown(
        f"""
        <div class="nav-rail">
            <div class="nav-brand">
                <div class="nav-logo">🧬</div>
                <div class="nav-title">Bio<span>Flow</span></div>
            </div>
            <div class="nav-section">Navigation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    label_map = {key: f"{icon} {label}" for key, icon, label in nav_items}
    options = [item[0] for item in nav_items]

    selected = st.radio(
        "Navigation",
        options=options,
        index=options.index(active_page),
        format_func=lambda x: label_map.get(x, x),
        key="nav_radio",
        label_visibility="collapsed",
    )

    return selected


# === Page Structure ===

def page_header(title: str, subtitle: str = "", icon: str = ""):
    """Page header with title and optional subtitle."""
    header_html = f"""
    <div style="margin-bottom: 2rem;">
        <h1 style="display: flex; align-items: center; gap: 0.75rem; margin: 0;">
            {f'<span style="font-size: 2rem;">{icon}</span>' if icon else ''}
            {title}
        </h1>
        {f'<p style="margin-top: 0.5rem; font-size: 1rem; color: {COLORS.text_muted};">{subtitle}</p>' if subtitle else ''}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def section_header(title: str, icon: str = "", link_text: str = "", link_action: Optional[Callable] = None):
    """Section header with optional action link."""
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown(f"""
        <div class="section-title">
            {f'<span>{icon}</span>' if icon else ''}
            {title}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if link_text:
            if st.button(link_text, key=f"section_{title}", use_container_width=True):
                if link_action:
                    link_action()


def divider():
    """Visual divider."""
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def spacer(height: str = "1rem"):
    """Vertical spacer."""
    st.markdown(f'<div style="height: {height};"></div>', unsafe_allow_html=True)


# === Metrics ===

def metric_card(
    value: str,
    label: str,
    icon: str = "📊",
    change: Optional[str] = None,
    change_type: str = "up",
    color: str = COLORS.primary
):
    """Single metric card with icon and optional trend."""
    bg_color = color.replace(")", ", 0.15)").replace("rgb", "rgba") if "rgb" in color else f"{color}22"
    change_html = ""
    if change:
        arrow = "↑" if change_type == "up" else "↓"
        change_html = f'<div class="metric-change {change_type}">{arrow} {change}</div>'
    
    st.markdown(f"""
    <div class="metric">
        <div class="metric-icon" style="background: {bg_color}; color: {color};">
            {icon}
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {change_html}
    </div>
    """, unsafe_allow_html=True)


def metric_row(metrics: List[Dict[str, Any]]):
    """Row of metric cards."""
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            metric_card(**metric)


# === Quick Actions ===

def quick_action(icon: str, title: str, description: str, key: str) -> bool:
    """Single quick action card. Returns True if clicked."""
    clicked = st.button(
        f"{icon}  {title}",
        key=key,
        use_container_width=True,
        help=description
    )
    return clicked


def quick_actions_grid(actions: List[Dict[str, Any]], columns: int = 4) -> Optional[str]:
    """Grid of quick action cards. Returns clicked action key or None."""
    cols = st.columns(columns)
    clicked_key = None
    
    for i, action in enumerate(actions):
        with cols[i % columns]:
            st.markdown(f"""
            <div class="quick-action">
                <span class="quick-action-icon">{action['icon']}</span>
                <div class="quick-action-title">{action['title']}</div>
                <div class="quick-action-desc">{action.get('description', '')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Select", key=action['key'], use_container_width=True):
                clicked_key = action['key']
    
    return clicked_key


# === Pipeline Progress ===

def pipeline_progress(steps: List[Dict[str, Any]]):
    """Visual pipeline with steps showing progress."""
    html = '<div class="pipeline">'
    
    for i, step in enumerate(steps):
        status = step.get('status', 'pending')
        icon = step.get('icon', str(i + 1))
        name = step.get('name', f'Step {i + 1}')
        
        # Display icon for completed steps
        if status == 'done':
            display = '✓'
        elif status == 'active':
            display = icon
        else:
            display = str(i + 1)
        
        html += f'''
        <div class="step">
            <div class="step-dot {status}">{display}</div>
            <span class="step-name">{name}</span>
        </div>
        '''
        
        # Add connecting line (except after last step)
        if i < len(steps) - 1:
            line_status = 'done' if status == 'done' else ''
            html += f'<div class="step-line {line_status}"></div>'
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# === Results ===

def result_card(
    title: str,
    score: float,
    properties: Dict[str, str] = None,
    badges: List[str] = None,
    key: str = ""
) -> bool:
    """Result card with score and properties. Returns True if clicked."""
    
    # Score color
    if score >= 0.8:
        score_class = "score-high"
    elif score >= 0.5:
        score_class = "score-med"
    else:
        score_class = "score-low"
    
    # Properties HTML
    props_html = ""
    if properties:
        props_html = '<div style="display: flex; gap: 1rem; margin-top: 0.75rem; flex-wrap: wrap;">'
        for k, v in properties.items():
            props_html += f'''
            <div style="font-size: 0.8125rem;">
                <span style="color: {COLORS.text_muted};">{k}:</span>
                <span style="color: {COLORS.text_secondary}; margin-left: 0.25rem;">{v}</span>
            </div>
            '''
        props_html += '</div>'
    
    # Badges HTML
    badges_html = ""
    if badges:
        badges_html = '<div style="display: flex; gap: 0.5rem; margin-top: 0.75rem;">'
        for b in badges:
            badges_html += f'<span class="badge badge-primary">{b}</span>'
        badges_html += '</div>'
    
    st.markdown(f"""
    <div class="result">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="font-weight: 600; color: {COLORS.text_primary};">{title}</div>
            <div class="{score_class}" style="font-size: 1.25rem; font-weight: 700;">{score:.1%}</div>
        </div>
        {props_html}
        {badges_html}
    </div>
    """, unsafe_allow_html=True)
    
    return st.button("View Details", key=key, use_container_width=True) if key else False


def results_list(results: List[Dict[str, Any]], empty_message: str = "No results found"):
    """List of result cards."""
    if not results:
        empty_state(icon="🔍", title="No Results", description=empty_message)
        return
    
    for i, result in enumerate(results):
        result_card(
            title=result.get('title', f'Result {i + 1}'),
            score=result.get('score', 0),
            properties=result.get('properties'),
            badges=result.get('badges'),
            key=f"result_{i}"
        )
        spacer("0.75rem")


# === Charts ===

def bar_chart(data: Dict[str, float], title: str = "", height: int = 300):
    """Styled bar chart."""
    fig = go.Figure(data=[
        go.Bar(
            x=list(data.keys()),
            y=list(data.values()),
            marker_color=COLORS.primary,
            marker_line_width=0,
        )
    ])
    
    fig.update_layout(
        title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color=COLORS.text_secondary),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor=COLORS.border,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS.border,
            showline=False,
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def scatter_chart(x: List, y: List, labels: List = None, title: str = "", height: int = 400):
    """Styled scatter plot."""
    fig = go.Figure(data=[
        go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=10,
                color=COLORS.primary,
                opacity=0.7,
            ),
            text=labels,
            hovertemplate='<b>%{text}</b><br>X: %{x}<br>Y: %{y}<extra></extra>' if labels else None,
        )
    ])
    
    fig.update_layout(
        title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color=COLORS.text_secondary),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS.border,
            showline=True,
            linecolor=COLORS.border,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS.border,
            showline=True,
            linecolor=COLORS.border,
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def heatmap(data: List[List[float]], x_labels: List[str], y_labels: List[str], title: str = "", height: int = 400):
    """Styled heatmap."""
    fig = go.Figure(data=[
        go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale=[
                [0, COLORS.bg_hover],
                [0.5, COLORS.primary],
                [1, COLORS.cyan],
            ],
        )
    ])
    
    fig.update_layout(
        title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color=COLORS.text_secondary),
        height=height,
        margin=dict(l=80, r=20, t=40, b=60),
    )
    
    st.plotly_chart(fig, use_container_width=True)


# === Data Display ===

def data_table(data: List[Dict], columns: List[str] = None):
    """Styled data table."""
    import pandas as pd
    df = pd.DataFrame(data)
    if columns:
        df = df[columns]
    st.dataframe(df, use_container_width=True, hide_index=True)


# === States ===

def empty_state(icon: str = "📭", title: str = "No Data", description: str = ""):
    """Empty state placeholder."""
    st.markdown(f"""
    <div class="empty">
        <div class="empty-icon">{icon}</div>
        <div class="empty-title">{title}</div>
        <div class="empty-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def loading_state(message: str = "Loading..."):
    """Loading state with spinner."""
    st.markdown(f"""
    <div class="loading">
        <div class="spinner"></div>
        <div class="loading-text">{message}</div>
    </div>
    """, unsafe_allow_html=True)


# === Molecule Display ===

def molecule_2d(smiles: str, size: int = 200):
    """Display 2D molecule structure from SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        import base64
        from io import BytesIO
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(size, size))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            st.markdown(f"""
            <div class="mol-container">
                <img src="data:image/png;base64,{img_str}" alt="Molecule" style="max-width: 100%; height: auto;">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Invalid SMILES")
    except ImportError:
        st.info(f"SMILES: `{smiles}`")


# === Evidence & Links ===

def evidence_row(items: List[Dict[str, str]]):
    """Row of evidence/source links."""
    html = '<div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.75rem;">'
    for item in items:
        icon = item.get('icon', '📄')
        label = item.get('label', 'Source')
        url = item.get('url', '#')
        html += f'''
        <a href="{url}" target="_blank" class="evidence">
            <span>{icon}</span>
            <span>{label}</span>
        </a>
        '''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# === Badges ===

def badge(text: str, variant: str = "primary"):
    """Inline badge component."""
    st.markdown(f'<span class="badge badge-{variant}">{text}</span>', unsafe_allow_html=True)


def badge_row(badges: List[Dict[str, str]]):
    """Row of badges."""
    html = '<div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">'
    for b in badges:
        text = b.get('text', '')
        variant = b.get('variant', 'primary')
        html += f'<span class="badge badge-{variant}">{text}</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
