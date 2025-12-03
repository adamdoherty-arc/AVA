"""
Centralized Page Registry for Magnus Trading Platform
=====================================================

Replaces magic string page routing with a type-safe registry system.
All pages are registered here with metadata for navigation and discovery.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Literal
from enum import Enum


# ============================================================================
# PAGE CATEGORIES
# ============================================================================

class PageCategory(str, Enum):
    """Categories for organizing pages in navigation"""
    DASHBOARD = "dashboard"
    TRADING = "trading"
    OPTIONS = "options"
    SPORTS_BETTING = "sports_betting"
    ANALYTICS = "analytics"
    AI_TOOLS = "ai_tools"
    SYSTEM = "system"
    ADMIN = "admin"


# ============================================================================
# PAGE METADATA
# ============================================================================

@dataclass
class PageMetadata:
    """Metadata for a registered page"""

    # Identification
    page_id: str  # Unique identifier (e.g., "dashboard", "ava-chatbot")
    display_name: str  # Display name for navigation (e.g., "Dashboard", "AVA Chatbot")
    icon: str  # Emoji icon for the page

    # Navigation
    category: PageCategory  # Category for grouping in navigation
    file_path: str  # Relative path to the page file (e.g., "dashboard.py")

    # Features
    description: str  # Brief description of the page
    keywords: List[str]  # Keywords for search/discovery
    requires_auth: bool = False  # Whether authentication is required
    requires_setup: bool = False  # Whether initial setup is needed
    is_active: bool = True  # Whether the page is currently active

    # Order and visibility
    sort_order: int = 100  # Order within category (lower = first)
    show_in_nav: bool = True  # Whether to show in navigation menu

    # Dependencies
    required_env_vars: List[str] = None  # Required environment variables
    required_services: List[str] = None  # Required external services

    def __post_init__(self) -> None:
        """Initialize default values"""
        if self.required_env_vars is None:
            self.required_env_vars = []
        if self.required_services is None:
            self.required_services = []


# ============================================================================
# PAGE REGISTRY
# ============================================================================

class PageRegistry:
    """Central registry of all pages in the application"""

    def __init__(self) -> None:
        self._pages: Dict[str, PageMetadata] = {}
        self._initialize_pages()

    def _initialize_pages(self) -> None:
        """Initialize all page registrations"""

        # ====================================================================
        # DASHBOARD PAGES
        # ====================================================================

        self.register(PageMetadata(
            page_id="dashboard",
            display_name="Dashboard",
            icon="🏠",
            category=PageCategory.DASHBOARD,
            file_path="dashboard.py",
            description="Main dashboard with portfolio overview and quick actions",
            keywords=["home", "overview", "summary", "main"],
            sort_order=1
        ))

        # ====================================================================
        # AI TOOLS
        # ====================================================================

        self.register(PageMetadata(
            page_id="ava-chatbot",
            display_name="AVA Chatbot",
            icon="🤖",
            category=PageCategory.AI_TOOLS,
            file_path="ava_chatbot_page.py",
            description="AI assistant with 33+ specialized agents for trading and analysis",
            keywords=["ai", "assistant", "chat", "ava", "agents"],
            sort_order=1
        ))

        self.register(PageMetadata(
            page_id="ai-options-agent",
            display_name="AI Options Advisor",
            icon="🧠",
            category=PageCategory.AI_TOOLS,
            file_path="ai_options_agent_page.py",
            description="AI-powered options strategy recommendations",
            keywords=["ai", "options", "advisor", "recommendations"],
            sort_order=2
        ))

        # ====================================================================
        # TRADING PAGES
        # ====================================================================

        self.register(PageMetadata(
            page_id="positions",
            display_name="Positions",
            icon="📊",
            category=PageCategory.TRADING,
            file_path="positions_page_improved.py",
            description="Live portfolio positions with Robinhood integration",
            keywords=["positions", "portfolio", "holdings", "robinhood"],
            requires_setup=True,
            required_env_vars=["ROBINHOOD_USERNAME", "ROBINHOOD_PASSWORD"],
            sort_order=1
        ))

        self.register(PageMetadata(
            page_id="xtrades-watchlists",
            display_name="XTrades Watchlists",
            icon="📱",
            category=PageCategory.TRADING,
            file_path="xtrades_watchlists_page.py",
            description="Discord alerts and watchlist management from XTrades",
            keywords=["discord", "alerts", "signals", "xtrades"],
            requires_setup=True,
            required_env_vars=["DISCORD_TOKEN"],
            sort_order=2
        ))

        # ====================================================================
        # OPTIONS TRADING
        # ====================================================================

        self.register(PageMetadata(
            page_id="options-analysis",
            display_name="Options Analysis",
            icon="📈",
            category=PageCategory.OPTIONS,
            file_path="options_analysis_page.py",
            description="Comprehensive options chain analysis and Greeks",
            keywords=["options", "analysis", "greeks", "chain"],
            sort_order=1
        ))

        self.register(PageMetadata(
            page_id="calendar-spreads",
            display_name="Calendar Spreads",
            icon="📅",
            category=PageCategory.OPTIONS,
            file_path="calendar_spreads_page.py",
            description="Calendar spread opportunities with AI analysis",
            keywords=["calendar", "spreads", "theta", "options"],
            sort_order=2
        ))

        self.register(PageMetadata(
            page_id="seven-day-dte",
            display_name="7-Day DTE Scanner",
            icon="⏰",
            category=PageCategory.OPTIONS,
            file_path="seven_day_dte_scanner_page.py",
            description="High theta capture opportunities with 0-7 DTE",
            keywords=["dte", "scanner", "theta", "weekly"],
            sort_order=3
        ))

        self.register(PageMetadata(
            page_id="premium-scanner",
            display_name="Premium Scanner",
            icon="💰",
            category=PageCategory.OPTIONS,
            file_path="premium_scanner_page.py",
            description="Premium analysis with Greeks and IV metrics",
            keywords=["premium", "scanner", "iv", "greeks"],
            sort_order=4
        ))

        self.register(PageMetadata(
            page_id="earnings-calendar",
            display_name="Earnings Calendar",
            icon="📆",
            category=PageCategory.OPTIONS,
            file_path="earnings_calendar_page.py",
            description="Earnings tracking and avoidance for safer trades",
            keywords=["earnings", "calendar", "events"],
            sort_order=5
        ))

        # ====================================================================
        # SPORTS BETTING
        # ====================================================================

        self.register(PageMetadata(
            page_id="game-cards",
            display_name="Game Cards",
            icon="🏟️",
            category=PageCategory.SPORTS_BETTING,
            file_path="game_cards_visual_page.py",
            description="Live sports game monitoring with Kalshi integration",
            keywords=["sports", "games", "kalshi", "live", "betting"],
            requires_setup=True,
            required_env_vars=["KALSHI_EMAIL", "KALSHI_PASSWORD"],
            sort_order=1
        ))

        self.register(PageMetadata(
            page_id="prediction-markets",
            display_name="Prediction Markets",
            icon="🎯",
            category=PageCategory.SPORTS_BETTING,
            file_path="prediction_markets_page.py",
            description="Kalshi prediction market analysis and opportunities",
            keywords=["kalshi", "prediction", "markets", "sports"],
            requires_setup=True,
            required_env_vars=["KALSHI_EMAIL", "KALSHI_PASSWORD"],
            sort_order=2
        ))

        self.register(PageMetadata(
            page_id="kalshi-nfl",
            display_name="Kalshi NFL Markets",
            icon="🏈",
            category=PageCategory.SPORTS_BETTING,
            file_path="kalshi_nfl_markets_page.py",
            description="NFL-specific prediction markets on Kalshi",
            keywords=["nfl", "football", "kalshi", "markets"],
            requires_setup=True,
            required_env_vars=["KALSHI_EMAIL", "KALSHI_PASSWORD"],
            sort_order=3
        ))

        # ====================================================================
        # ANALYTICS
        # ====================================================================

        self.register(PageMetadata(
            page_id="technical-indicators",
            display_name="Technical Indicators",
            icon="📉",
            category=PageCategory.ANALYTICS,
            file_path="technical_indicators_page.py",
            description="Advanced technical analysis and charting",
            keywords=["technical", "indicators", "charts", "analysis"],
            sort_order=1
        ))

        self.register(PageMetadata(
            page_id="supply-demand-zones",
            display_name="Supply/Demand Zones",
            icon="🎚️",
            category=PageCategory.ANALYTICS,
            file_path="supply_demand_zones_page.py",
            description="Price action zones and key levels",
            keywords=["supply", "demand", "zones", "levels"],
            sort_order=2
        ))

        self.register(PageMetadata(
            page_id="sector-analysis",
            display_name="Sector Analysis",
            icon="🏭",
            category=PageCategory.ANALYTICS,
            file_path="sector_analysis_page.py",
            description="Sector rotation and market breadth analysis",
            keywords=["sector", "rotation", "market", "breadth"],
            sort_order=3
        ))

        # ====================================================================
        # SYSTEM PAGES
        # ====================================================================

        self.register(PageMetadata(
            page_id="agent-management",
            display_name="Agent Management",
            icon="⚙️",
            category=PageCategory.SYSTEM,
            file_path="agent_management_page.py",
            description="Manage and configure AI agents",
            keywords=["agents", "config", "management"],
            sort_order=1
        ))

        self.register(PageMetadata(
            page_id="enhancement-qa",
            display_name="Enhancement & QA",
            icon="🎯",
            category=PageCategory.SYSTEM,
            file_path="enhancement_qa_management_page.py",
            description="Enhancement tracking and QA management",
            keywords=["qa", "enhancement", "tasks", "management"],
            sort_order=2
        ))

        self.register(PageMetadata(
            page_id="health-dashboard",
            display_name="Health Dashboard",
            icon="🏥",
            category=PageCategory.SYSTEM,
            file_path="health_dashboard_page.py",
            description="System health monitoring and diagnostics",
            keywords=["health", "monitoring", "diagnostics", "status"],
            sort_order=3
        ))

        self.register(PageMetadata(
            page_id="analytics-performance",
            display_name="Analytics Performance",
            icon="⚡",
            category=PageCategory.SYSTEM,
            file_path="analytics_performance_page.py",
            description="Performance metrics and optimization",
            keywords=["performance", "metrics", "optimization"],
            sort_order=4
        ))

        # ====================================================================
        # DEMO/TESTING PAGES
        # ====================================================================

        self.register(PageMetadata(
            page_id="ui-theme-demo",
            display_name="UI Theme Demo",
            icon="🎨",
            category=PageCategory.SYSTEM,
            file_path="ui_theme_demo_page.py",
            description="Demonstration of unified UI theme system",
            keywords=["theme", "ui", "demo", "components"],
            show_in_nav=True,
            sort_order=99
        ))

    def register(self, page: PageMetadata) -> None:
        """Register a page in the registry"""
        if page.page_id in self._pages:
            raise ValueError(f"Page '{page.page_id}' is already registered")
        self._pages[page.page_id] = page

    def get_page(self, page_id: str) -> Optional[PageMetadata]:
        """Get a page by ID"""
        return self._pages.get(page_id)

    def get_all_pages(self) -> List[PageMetadata]:
        """Get all registered pages"""
        return list(self._pages.values())

    def get_pages_by_category(self, category: PageCategory) -> List[PageMetadata]:
        """Get all pages in a category, sorted by sort_order"""
        pages = [p for p in self._pages.values() if p.category == category]
        return sorted(pages, key=lambda p: p.sort_order)

    def get_active_pages(self) -> List[PageMetadata]:
        """Get all active pages"""
        return [p for p in self._pages.values() if p.is_active]

    def get_nav_pages(self) -> List[PageMetadata]:
        """Get pages that should show in navigation"""
        return [p for p in self._pages.values() if p.show_in_nav and p.is_active]

    def search_pages(self, query: str) -> List[PageMetadata]:
        """Search pages by name, description, or keywords"""
        query = query.lower()
        results = []
        for page in self._pages.values():
            if (query in page.display_name.lower() or
                query in page.description.lower() or
                any(query in keyword for keyword in page.keywords)):
                results.append(page)
        return results

    def get_navigation_structure(self) -> Dict[PageCategory, List[PageMetadata]]:
        """Get pages organized by category for navigation menus"""
        structure = {}
        for category in PageCategory:
            pages = self.get_pages_by_category(category)
            pages = [p for p in pages if p.show_in_nav and p.is_active]
            if pages:  # Only include categories with pages
                structure[category] = pages
        return structure

    def validate_page_setup(self, page_id: str) -> tuple[bool, List[str]]:
        """
        Validate if a page's requirements are met

        Returns:
            (is_valid, list_of_missing_requirements)
        """
        import os

        page = self.get_page(page_id)
        if not page:
            return False, [f"Page '{page_id}' not found"]

        missing = []

        # Check environment variables
        for env_var in page.required_env_vars:
            if not os.getenv(env_var):
                missing.append(f"Missing environment variable: {env_var}")

        # Check file exists
        from pathlib import Path
        page_path = Path(page.file_path)
        if not page_path.exists():
            missing.append(f"Page file not found: {page.file_path}")

        return len(missing) == 0, missing


# ============================================================================
# GLOBAL REGISTRY INSTANCE
# ============================================================================

_registry = PageRegistry()


def get_registry() -> PageRegistry:
    """Get the global page registry instance"""
    return _registry


# ============================================================================
# NAVIGATION HELPERS
# ============================================================================

def render_sidebar_navigation():
    """Render navigation in Streamlit sidebar"""
    import streamlit as st

    registry = get_registry()
    structure = registry.get_navigation_structure()

    st.sidebar.title("Navigation")

    for category, pages in structure.items():
        # Category header
        st.sidebar.markdown(f"### {category.value.replace('_', ' ').title()}")

        # Pages in category
        for page in pages:
            if st.sidebar.button(
                f"{page.icon} {page.display_name}",
                key=f"nav_{page.page_id}",
                use_container_width=True
            ):
                # Navigate to page (implementation depends on your routing)
                st.session_state.current_page = page.page_id
                st.rerun()


def get_page_url(page_id: str) -> str:
    """Get the URL for a page"""
    page = get_registry().get_page(page_id)
    if not page:
        return "#"
    return f"/{page.file_path.replace('.py', '')}"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Get all pages
    registry = get_registry()

    print("All registered pages:")
    for page in registry.get_all_pages():
        print(f"  {page.icon} {page.display_name} ({page.page_id})")

    print("\nPages by category:")
    for category, pages in registry.get_navigation_structure().items():
        print(f"\n{category.value.upper()}:")
        for page in pages:
            print(f"  {page.icon} {page.display_name}")

    print("\nSearch for 'options':")
    results = registry.search_pages("options")
    for page in results:
        print(f"  {page.icon} {page.display_name} - {page.description}")

    print("\nValidate page setup:")
    is_valid, missing = registry.validate_page_setup("positions")
    if is_valid:
        print("  ✅ All requirements met")
    else:
        print("  ❌ Missing requirements:")
        for item in missing:
            print(f"    - {item}")
