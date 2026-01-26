"""
BioFlow UI - Modern Design System
==================================
Clean, minimal, and highly usable interface.
"""

from dataclasses import dataclass


@dataclass
class Colors:
    """Color palette - Modern dark theme."""
    # Primary
    primary: str = "#8B5CF6"
    primary_hover: str = "#A78BFA"
    primary_muted: str = "rgba(139, 92, 246, 0.15)"
    
    # Accents
    cyan: str = "#22D3EE"
    emerald: str = "#34D399"
    amber: str = "#FBBF24"
    rose: str = "#FB7185"
    
    # Backgrounds
    bg_app: str = "#0C0E14"
    bg_surface: str = "#14161E"
    bg_elevated: str = "#1C1F2B"
    bg_hover: str = "#252836"
    
    # Text
    text_primary: str = "#F8FAFC"
    text_secondary: str = "#A1A7BB"
    text_muted: str = "#6B7280"
    
    # Borders
    border: str = "#2A2D3A"
    border_hover: str = "#3F4354"
    
    # Status
    success: str = "#10B981"
    warning: str = "#F59E0B"
    error: str = "#EF4444"
    info: str = "#3B82F6"


COLORS = Colors()


def get_css() -> str:
    """Minimalist, professional CSS using string concatenation to avoid f-string issues."""
    
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --primary: """ + COLORS.primary + """;
        --bg-app: """ + COLORS.bg_app + """;
        --bg-surface: """ + COLORS.bg_surface + """;
        --text: """ + COLORS.text_primary + """;
        --text-muted: """ + COLORS.text_muted + """;
        --border: """ + COLORS.border + """;
        --radius: 12px;
        --transition: 150ms ease;
    }
    
    .stApp {
        background: """ + COLORS.bg_app + """;
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: """ + COLORS.border + """; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: """ + COLORS.border_hover + """; }
    
    section[data-testid="stSidebar"] { display: none !important; }
    
    h1, h2, h3 {
        font-weight: 600;
        color: """ + COLORS.text_primary + """;
        letter-spacing: -0.025em;
    }
    
    h1 { font-size: 1.875rem; margin-bottom: 0.5rem; }
    h2 { font-size: 1.5rem; }
    h3 { font-size: 1.125rem; }
    
    p { color: """ + COLORS.text_secondary + """; line-height: 1.6; }
    
    .card {
        background: """ + COLORS.bg_surface + """;
        border: 1px solid """ + COLORS.border + """;
        border-radius: var(--radius);
        padding: 1.25rem;
    }
    
    .metric {
        background: """ + COLORS.bg_surface + """;
        border: 1px solid """ + COLORS.border + """;
        border-radius: var(--radius);
        padding: 1.25rem;
        transition: border-color var(--transition);
    }
    
    .metric:hover { border-color: """ + COLORS.primary + """; }
    
    .metric-icon {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.375rem;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: """ + COLORS.text_primary + """;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: """ + COLORS.text_muted + """;
        margin-top: 0.375rem;
    }
    
    .metric-change {
        display: inline-flex;
        align-items: center;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        margin-top: 0.5rem;
    }
    
    .metric-change.up { background: rgba(16, 185, 129, 0.15); color: """ + COLORS.success + """; }
    .metric-change.down { background: rgba(239, 68, 68, 0.15); color: """ + COLORS.error + """; }
    
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.875rem;
        border-radius: 8px;
        padding: 0.625rem 1.25rem;
        transition: all var(--transition);
        border: none;
    }
    
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox > div > div {
        background: """ + COLORS.bg_app + """ !important;
        border: 1px solid """ + COLORS.border + """ !important;
        border-radius: 10px !important;
        color: """ + COLORS.text_primary + """ !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: """ + COLORS.primary + """ !important;
        box-shadow: 0 0 0 3px """ + COLORS.primary_muted + """ !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: """ + COLORS.bg_surface + """;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid """ + COLORS.border + """;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 0.625rem 1.25rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.875rem;
        color: """ + COLORS.text_muted + """;
        background: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: """ + COLORS.primary + """ !important;
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none; }
    
    .pipeline {
        display: flex;
        align-items: center;
        background: """ + COLORS.bg_surface + """;
        border: 1px solid """ + COLORS.border + """;
        border-radius: var(--radius);
        padding: 1.5rem;
        gap: 0;
    }
    
    .step {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
        flex: 1;
    }
    
    .step-dot {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.125rem;
        font-weight: 600;
        transition: all var(--transition);
    }
    
    .step-dot.pending {
        background: """ + COLORS.bg_hover + """;
        color: """ + COLORS.text_muted + """;
        border: 2px dashed """ + COLORS.border_hover + """;
    }
    
    .step-dot.active {
        background: """ + COLORS.primary + """;
        color: white;
        box-shadow: 0 0 24px rgba(139, 92, 246, 0.5);
    }
    
    .step-dot.done {
        background: """ + COLORS.emerald + """;
        color: white;
    }
    
    .step-name {
        font-size: 0.75rem;
        font-weight: 500;
        color: """ + COLORS.text_muted + """;
    }
    
    .step-line {
        flex: 0.6;
        height: 2px;
        background: """ + COLORS.border + """;
    }
    
    .step-line.done { background: """ + COLORS.emerald + """; }
    
    .result {
        background: """ + COLORS.bg_surface + """;
        border: 1px solid """ + COLORS.border + """;
        border-radius: var(--radius);
        padding: 1.25rem;
        transition: all var(--transition);
        cursor: pointer;
    }
    
    .result:hover {
        border-color: """ + COLORS.primary + """;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    
    .score-high { color: """ + COLORS.emerald + """; }
    .score-med { color: """ + COLORS.amber + """; }
    .score-low { color: """ + COLORS.rose + """; }
    
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.625rem;
        border-radius: 6px;
        font-size: 0.6875rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .badge-primary { background: """ + COLORS.primary_muted + """; color: """ + COLORS.primary + """; }
    .badge-success { background: rgba(16, 185, 129, 0.15); color: """ + COLORS.success + """; }
    .badge-warning { background: rgba(245, 158, 11, 0.15); color: """ + COLORS.warning + """; }
    .badge-error { background: rgba(239, 68, 68, 0.15); color: """ + COLORS.error + """; }
    
    .quick-action {
        background: """ + COLORS.bg_surface + """;
        border: 1px solid """ + COLORS.border + """;
        border-radius: var(--radius);
        padding: 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all var(--transition);
    }
    
    .quick-action:hover {
        border-color: """ + COLORS.primary + """;
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
    }
    
    .quick-action-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        display: block;
    }
    
    .quick-action-title {
        font-size: 0.9375rem;
        font-weight: 600;
        color: """ + COLORS.text_primary + """;
    }
    
    .quick-action-desc {
        font-size: 0.8125rem;
        color: """ + COLORS.text_muted + """;
        margin-top: 0.25rem;
    }
    
    .section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: """ + COLORS.text_primary + """;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-link {
        font-size: 0.8125rem;
        color: """ + COLORS.primary + """;
        cursor: pointer;
    }
    
    .section-link:hover { text-decoration: underline; }
    
    .empty {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem 2rem;
        text-align: center;
    }
    
    .empty-icon { font-size: 3.5rem; margin-bottom: 1rem; opacity: 0.4; }
    .empty-title { font-size: 1.125rem; font-weight: 600; color: """ + COLORS.text_primary + """; }
    .empty-desc { font-size: 0.9375rem; color: """ + COLORS.text_muted + """; max-width: 320px; margin-top: 0.5rem; }
    
    .loading {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 3rem;
    }
    
    .spinner {
        width: 40px;
        height: 40px;
        border: 3px solid """ + COLORS.border + """;
        border-top-color: """ + COLORS.primary + """;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    
    @keyframes spin { to { transform: rotate(360deg); } }
    
    .loading-text {
        margin-top: 1rem;
        color: """ + COLORS.text_muted + """;
        font-size: 0.875rem;
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, """ + COLORS.primary + """ 0%, """ + COLORS.cyan + """ 100%);
        border-radius: 4px;
    }
    
    .stProgress > div > div {
        background: """ + COLORS.bg_hover + """;
        border-radius: 4px;
    }
    
    .divider {
        height: 1px;
        background: """ + COLORS.border + """;
        margin: 1.5rem 0;
    }
    
    .mol-container {
        background: white;
        border-radius: 10px;
        padding: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .evidence {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.5rem 0.75rem;
        background: """ + COLORS.bg_app + """;
        border: 1px solid """ + COLORS.border + """;
        border-radius: 8px;
        font-size: 0.8125rem;
        color: """ + COLORS.text_secondary + """;
        transition: all var(--transition);
        text-decoration: none;
    }
    
    .evidence:hover {
        border-color: """ + COLORS.primary + """;
        color: """ + COLORS.primary + """;
    }
    
    .stAlert { border-radius: 10px; border: none; }
    
    .stDataFrame {
        border-radius: var(--radius);
        overflow: hidden;
        border: 1px solid """ + COLORS.border + """;
    }

    .block-container {
        padding-top: 1.25rem;
    }

    .nav-rail {
        position: sticky;
        top: 1rem;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        padding: 1rem;
        background: """ + COLORS.bg_surface + """;
        border: 1px solid """ + COLORS.border + """;
        border-radius: 16px;
        margin-bottom: 1rem;
    }

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid """ + COLORS.border + """;
    }

    .nav-logo { font-size: 1.5rem; }

    .nav-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: """ + COLORS.text_primary + """;
    }

    .nav-title span {
        background: linear-gradient(135deg, """ + COLORS.primary + """ 0%, """ + COLORS.cyan + """ 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .nav-section {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: """ + COLORS.text_muted + """;
    }

    div[data-testid="stRadio"] {
        background: """ + COLORS.bg_surface + """;
        border: 1px solid """ + COLORS.border + """;
        border-radius: 16px;
        padding: 0.75rem;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        margin-top: 0.25rem;
    }

    div[data-testid="stRadio"] input {
        display: none !important;
    }

    div[data-testid="stRadio"] label {
        background: """ + COLORS.bg_app + """;
        border: 1px solid """ + COLORS.border + """;
        border-radius: 12px;
        padding: 0.65rem 0.9rem;
        font-weight: 500;
        color: """ + COLORS.text_secondary + """;
        transition: all var(--transition);
        margin: 0 !important;
    }

    div[data-testid="stRadio"] label:hover {
        border-color: """ + COLORS.primary + """;
        color: """ + COLORS.text_primary + """;
    }

    div[data-testid="stRadio"] label:has(input:checked) {
        background: """ + COLORS.primary + """;
        border-color: """ + COLORS.primary + """;
        color: white;
        box-shadow: 0 8px 20px rgba(139, 92, 246, 0.25);
    }

    .hero {
        position: relative;
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.12) 0%, rgba(34, 211, 238, 0.08) 100%);
        border: 1px solid """ + COLORS.border + """;
        border-radius: 20px;
        padding: 2.75rem;
        overflow: hidden;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: """ + COLORS.primary_muted + """;
        color: """ + COLORS.primary + """;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: """ + COLORS.text_primary + """;
        margin-top: 1rem;
        line-height: 1.1;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: """ + COLORS.text_muted + """;
        margin-top: 0.75rem;
        max-width: 560px;
    }

    .hero-actions {
        display: flex;
        gap: 0.75rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }

    .hero-card {
        background: """ + COLORS.bg_surface + """;
        border: 1px solid """ + COLORS.border + """;
        border-radius: 16px;
        padding: 1.5rem;
    }
    </style>
    """
    
    return css
