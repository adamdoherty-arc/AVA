"""
Unified UI Theme Configuration for Magnus Trading Platform
Provides consistent styling, colors, and components across all pages
"""
import streamlit as st
from typing import Dict, Optional, Literal
from dataclasses import dataclass

# ============================================================================
# COLOR PALETTE
# ============================================================================

@dataclass
class ColorPalette:
    """Centralized color palette for the entire application"""

    # Primary Brand Colors
    primary: str = "#4F46E5"  # Indigo - Main brand color
    primary_dark: str = "#4338CA"
    primary_light: str = "#818CF8"

    # Secondary Colors
    secondary: str = "#10B981"  # Green - Success/Positive
    secondary_dark: str = "#059669"
    secondary_light: str = "#34D399"

    # Accent Colors
    accent: str = "#F59E0B"  # Amber - Warning/Attention
    accent_dark: str = "#D97706"
    accent_light: str = "#FBBF24"

    # Status Colors
    success: str = "#10B981"  # Green
    success_bg: str = "#D4EDDA"
    warning: str = "#F59E0B"  # Amber
    warning_bg: str = "#FFF3CD"
    error: str = "#EF4444"  # Red
    error_bg: str = "#F8D7DA"
    info: str = "#3B82F6"  # Blue
    info_bg: str = "#D1E7FF"

    # Live/Real-time indicators
    live: str = "#EF4444"  # Red for live indicator
    live_pulse: str = "#DC2626"

    # Neutral/Background Colors
    background: str = "#FFFFFF"
    background_secondary: str = "#F9FAFB"
    background_tertiary: str = "#F3F4F6"

    # Dark Mode Colors
    dark_bg: str = "#1F2937"
    dark_bg_secondary: str = "#111827"
    dark_bg_tertiary: str = "#374151"

    # Text Colors
    text_primary: str = "#111827"
    text_secondary: str = "#6B7280"
    text_tertiary: str = "#9CA3AF"
    text_inverse: str = "#FFFFFF"

    # Border Colors
    border: str = "#E5E7EB"
    border_dark: str = "#D1D5DB"
    border_light: str = "#F3F4F6"

    # Chart Colors (for plotly)
    chart_primary: str = "#4F46E5"
    chart_positive: str = "#10B981"
    chart_negative: str = "#EF4444"
    chart_neutral: str = "#6B7280"


# ============================================================================
# TYPOGRAPHY
# ============================================================================

@dataclass
class Typography:
    """Typography settings"""

    # Font Families
    font_primary: str = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif"
    font_monospace: str = "'SF Mono', Monaco, 'Cascadia Code', 'Courier New', monospace"

    # Font Sizes
    size_xs: str = "0.75rem"   # 12px
    size_sm: str = "0.875rem"  # 14px
    size_base: str = "1rem"     # 16px
    size_lg: str = "1.125rem"   # 18px
    size_xl: str = "1.25rem"    # 20px
    size_2xl: str = "1.5rem"    # 24px
    size_3xl: str = "1.875rem"  # 30px
    size_4xl: str = "2.25rem"   # 36px

    # Font Weights
    weight_normal: str = "400"
    weight_medium: str = "500"
    weight_semibold: str = "600"
    weight_bold: str = "700"

    # Line Heights
    line_height_tight: str = "1.25"
    line_height_normal: str = "1.5"
    line_height_relaxed: str = "1.75"


# ============================================================================
# SPACING
# ============================================================================

@dataclass
class Spacing:
    """Spacing/padding/margin constants"""

    xs: str = "0.25rem"   # 4px
    sm: str = "0.5rem"    # 8px
    base: str = "1rem"    # 16px
    md: str = "1.5rem"    # 24px
    lg: str = "2rem"      # 32px
    xl: str = "3rem"      # 48px
    xxl: str = "4rem"     # 64px


# ============================================================================
# THEME CONFIGURATION
# ============================================================================

class UITheme:
    """Main theme configuration class"""

    def __init__(self, mode: Literal["light", "dark"] = "light"):
        self.mode = mode
        self.colors = ColorPalette()
        self.typography = Typography()
        self.spacing = Spacing()

    def get_page_config(
        self,
        page_title: str,
        page_icon: str = "🤖",
        layout: Literal["centered", "wide"] = "wide",
        initial_sidebar_state: Literal["auto", "expanded", "collapsed"] = "auto"
    ) -> Dict:
        """Get standardized page configuration"""
        return {
            "page_title": page_title,
            "page_icon": page_icon,
            "layout": layout,
            "initial_sidebar_state": initial_sidebar_state,
            "menu_items": {
                'Get Help': 'https://github.com/yourusername/magnus',
                'Report a bug': 'https://github.com/yourusername/magnus/issues',
                'About': '# Magnus Trading Platform\nAI-powered trading analysis and automation'
            }
        }

    def apply_custom_css(self) -> None:
        """Apply custom CSS styling to the entire app"""
        css = f"""
        <style>
        /* ===== GLOBAL STYLES ===== */
        :root {{
            --primary-color: {self.colors.primary};
            --secondary-color: {self.colors.secondary};
            --background-color: {self.colors.background};
            --text-color: {self.colors.text_primary};
            --border-color: {self.colors.border};
        }}

        /* Font smoothing */
        * {{
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}

        /* ===== METRIC CARDS ===== */
        .stMetric {{
            background-color: {self.colors.background_secondary};
            padding: {self.spacing.base};
            border-radius: 8px;
            border: 1px solid {self.colors.border};
        }}

        .stMetric [data-testid="stMetricLabel"] {{
            color: {self.colors.text_secondary};
            font-size: {self.typography.size_sm};
            font-weight: {self.typography.weight_medium};
        }}

        .stMetric [data-testid="stMetricValue"] {{
            color: {self.colors.text_primary};
            font-size: {self.typography.size_2xl};
            font-weight: {self.typography.weight_bold};
        }}

        .stMetric [data-testid="stMetricDelta"] {{
            font-size: {self.typography.size_sm};
        }}

        /* ===== BUTTONS ===== */
        .stButton > button {{
            border-radius: 6px;
            font-weight: {self.typography.weight_medium};
            padding: {self.spacing.sm} {self.spacing.base};
            transition: all 0.2s ease;
            border: 1px solid {self.colors.border};
        }}

        .stButton > button:hover {{
            border-color: {self.colors.primary};
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .stButton > button[kind="primary"] {{
            background-color: {self.colors.primary};
            color: white;
            border: none;
        }}

        .stButton > button[kind="primary"]:hover {{
            background-color: {self.colors.primary_dark};
        }}

        .stButton > button[kind="secondary"] {{
            background-color: {self.colors.secondary};
            color: white;
            border: none;
        }}

        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"] {{
            background-color: {self.colors.background_secondary};
            border-right: 1px solid {self.colors.border};
        }}

        section[data-testid="stSidebar"] .stButton button {{
            background-color: transparent !important;
            border: 1px solid transparent;
            width: 100%;
            text-align: left;
            padding: {self.spacing.sm} {self.spacing.base};
            margin-bottom: {self.spacing.xs};
        }}

        section[data-testid="stSidebar"] .stButton button:hover {{
            background-color: rgba(79, 70, 229, 0.1) !important;
            border-color: {self.colors.primary};
        }}

        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab-list"] {{
            gap: {self.spacing.sm};
            border-bottom: 2px solid {self.colors.border};
        }}

        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            padding: 0 {self.spacing.base};
            border-radius: 6px 6px 0 0;
            font-weight: {self.typography.weight_medium};
        }}

        .stTabs [aria-selected="true"] {{
            background-color: {self.colors.primary};
            color: white;
        }}

        /* ===== DATA TABLES ===== */
        .dataframe {{
            border: 1px solid {self.colors.border} !important;
            border-radius: 8px;
            overflow: hidden;
        }}

        .dataframe thead tr th {{
            background-color: {self.colors.primary} !important;
            color: white !important;
            font-weight: {self.typography.weight_semibold};
            padding: {self.spacing.sm} !important;
        }}

        .dataframe tbody tr:nth-child(even) {{
            background-color: {self.colors.background_secondary};
        }}

        .dataframe tbody tr:hover {{
            background-color: {self.colors.background_tertiary};
        }}

        /* ===== INPUTS ===== */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div {{
            border-radius: 6px;
            border-color: {self.colors.border};
        }}

        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {{
            border-color: {self.colors.primary};
            box-shadow: 0 0 0 1px {self.colors.primary};
        }}

        /* ===== ALERTS/STATUS BOXES ===== */
        .success-box {{
            background-color: {self.colors.success_bg};
            border-left: 4px solid {self.colors.success};
            padding: {self.spacing.base};
            border-radius: 6px;
            margin: {self.spacing.sm} 0;
        }}

        .warning-box {{
            background-color: {self.colors.warning_bg};
            border-left: 4px solid {self.colors.warning};
            padding: {self.spacing.base};
            border-radius: 6px;
            margin: {self.spacing.sm} 0;
        }}

        .error-box {{
            background-color: {self.colors.error_bg};
            border-left: 4px solid {self.colors.error};
            padding: {self.spacing.base};
            border-radius: 6px;
            margin: {self.spacing.sm} 0;
        }}

        .info-box {{
            background-color: {self.colors.info_bg};
            border-left: 4px solid {self.colors.info};
            padding: {self.spacing.base};
            border-radius: 6px;
            margin: {self.spacing.sm} 0;
        }}

        /* ===== LIVE INDICATOR ===== */
        .live-indicator {{
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: {self.colors.live};
            border-radius: 50%;
            margin-right: {self.spacing.xs};
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
                box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
            }}
            50% {{
                opacity: 0.7;
                box-shadow: 0 0 0 6px rgba(239, 68, 68, 0);
            }}
        }}

        /* ===== CARDS ===== */
        .card {{
            background-color: {self.colors.background};
            border: 1px solid {self.colors.border};
            border-radius: 8px;
            padding: {self.spacing.base};
            margin-bottom: {self.spacing.base};
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }}

        .card:hover {{
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
            transition: all 0.2s ease;
        }}

        .card-header {{
            font-size: {self.typography.size_lg};
            font-weight: {self.typography.weight_semibold};
            color: {self.colors.text_primary};
            margin-bottom: {self.spacing.sm};
            padding-bottom: {self.spacing.sm};
            border-bottom: 1px solid {self.colors.border};
        }}

        /* ===== STICKY HEADER ===== */
        .sticky-header {{
            position: sticky;
            top: 0;
            background-color: {self.colors.background};
            z-index: 999;
            padding: {self.spacing.base} 0;
            border-bottom: 2px solid {self.colors.border};
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}

        /* ===== BADGES ===== */
        .badge {{
            display: inline-block;
            padding: {self.spacing.xs} {self.spacing.sm};
            border-radius: 12px;
            font-size: {self.typography.size_xs};
            font-weight: {self.typography.weight_semibold};
        }}

        .badge-success {{
            background-color: {self.colors.success_bg};
            color: {self.colors.success};
        }}

        .badge-warning {{
            background-color: {self.colors.warning_bg};
            color: {self.colors.warning};
        }}

        .badge-error {{
            background-color: {self.colors.error_bg};
            color: {self.colors.error};
        }}

        .badge-info {{
            background-color: {self.colors.info_bg};
            color: {self.colors.info};
        }}

        /* ===== LOADING SPINNER ===== */
        .stSpinner > div {{
            border-top-color: {self.colors.primary} !important;
        }}

        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: {self.colors.background_secondary};
        }}

        ::-webkit-scrollbar-thumb {{
            background: {self.colors.border_dark};
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: {self.colors.primary};
        }}

        /* ===== PLOTLY CHARTS ===== */
        .js-plotly-plot .plotly {{
            border-radius: 8px;
        }}

        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    def get_chart_template(self) -> Dict:
        """Get standardized Plotly chart template"""
        return {
            "layout": {
                "font": {
                    "family": self.typography.font_primary,
                    "color": self.colors.text_primary
                },
                "plot_bgcolor": self.colors.background,
                "paper_bgcolor": self.colors.background,
                "colorway": [
                    self.colors.chart_primary,
                    self.colors.chart_positive,
                    self.colors.chart_negative,
                    self.colors.accent,
                    self.colors.info,
                ],
                "xaxis": {
                    "gridcolor": self.colors.border,
                    "linecolor": self.colors.border_dark,
                },
                "yaxis": {
                    "gridcolor": self.colors.border,
                    "linecolor": self.colors.border_dark,
                },
                "title": {
                    "font": {
                        "size": 18,
                        "color": self.colors.text_primary
                    }
                },
                "hovermode": "closest",
                "hoverlabel": {
                    "bgcolor": self.colors.background,
                    "bordercolor": self.colors.border_dark,
                },
            }
        }


# ============================================================================
# COMPONENT HELPERS
# ============================================================================

class UIComponents:
    """Reusable UI component helpers"""

    def __init__(self, theme: UITheme):
        self.theme = theme

    def status_badge(self, status: str, label: str) -> str:
        """Create a status badge HTML"""
        badge_class = f"badge badge-{status}"
        return f'<span class="{badge_class}">{label}</span>'

    def success_box(self, message: str) -> None:
        """Display a success message box"""
        st.markdown(
            f'<div class="success-box">✅ {message}</div>',
            unsafe_allow_html=True
        )

    def warning_box(self, message: str) -> None:
        """Display a warning message box"""
        st.markdown(
            f'<div class="warning-box">⚠️ {message}</div>',
            unsafe_allow_html=True
        )

    def error_box(self, message: str) -> None:
        """Display an error message box"""
        st.markdown(
            f'<div class="error-box">❌ {message}</div>',
            unsafe_allow_html=True
        )

    def info_box(self, message: str) -> None:
        """Display an info message box"""
        st.markdown(
            f'<div class="info-box">ℹ️ {message}</div>',
            unsafe_allow_html=True
        )

    def card(self, title: str, content: str) -> None:
        """Display a card component"""
        st.markdown(
            f'''
            <div class="card">
                <div class="card-header">{title}</div>
                <div>{content}</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    def live_indicator(self, text: str = "LIVE") -> str:
        """Create a live indicator HTML"""
        return f'<span class="live-indicator"></span>{text}'


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Create global theme instance
_default_theme = UITheme(mode="light")

def get_theme() -> UITheme:
    """Get the default theme instance"""
    return _default_theme

def init_page(
    page_title: str,
    page_icon: str = "🤖",
    layout: Literal["centered", "wide"] = "wide",
    apply_theme: bool = True
) -> UITheme:
    """
    Initialize a page with standard configuration and theme

    Usage:
        theme = init_page("My Page", "📊", "wide")
    """
    theme = get_theme()

    # Set page config
    config = theme.get_page_config(page_title, page_icon, layout)
    st.set_page_config(**config)

    # Apply custom CSS
    if apply_theme:
        theme.apply_custom_css()

    return theme


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage in a page
    theme = init_page("Example Page", "🎨", "wide")
    components = UIComponents(theme)

    st.title("UI Theme Example")

    # Status boxes
    components.success_box("This is a success message")
    components.warning_box("This is a warning message")
    components.error_box("This is an error message")
    components.info_box("This is an info message")

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Value", "$1,234.56", "+5.2%")
    with col2:
        st.metric("Active Trades", "12", "-2")
    with col3:
        st.metric("Win Rate", "68%", "+3%")

    # Card
    components.card(
        "Example Card",
        "This is a card component with themed styling"
    )

    # Badges
    st.markdown(
        components.status_badge("success", "Active") + " " +
        components.status_badge("warning", "Pending") + " " +
        components.status_badge("error", "Failed"),
        unsafe_allow_html=True
    )

    # Live indicator
    st.markdown(
        components.live_indicator("LIVE TRADING"),
        unsafe_allow_html=True
    )
