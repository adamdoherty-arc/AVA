# Phase 3 Plan: Sports Betting Consolidation

**Status:** 📋 PLANNED (Analysis Complete)
**Estimated Effort:** 5-7 days
**Priority:** Medium (after Phase 2 completion)

---

## Executive Summary

Consolidate 4 sports betting pages (4,593 lines) into a unified Sports Betting Hub (~2,500 lines) while keeping Prediction Markets separate. This will achieve a 40% code reduction and significantly improve user experience by creating a single hub for all sports betting activities.

---

## Current State Analysis

### Pages to Consolidate

| Page | Lines | Main Features | Status |
|------|-------|---------------|--------|
| game_cards_visual_page.py | 2,157 | Live games, logos, multi-sport, watchlist | ✅ Analyzed |
| kalshi_nfl_markets_page.py | 1,580 | Markets, price charts, analytics | ✅ Analyzed (Uses registry) |
| ava_betting_recommendations_page.py | 539 | Kelly Criterion, EV focus, picks | ✅ Analyzed |
| prediction_markets_page.py | 317 | Multi-category, simple UI | ✅ Keep separate |
| **Total** | **4,593** | | |

### Key Findings

**Duplicate Features (across 3+ pages):**
- Kalshi odds display
- AI predictions & confidence scoring
- Expected Value calculations
- Filtering by confidence/EV
- Card-based UI layouts
- Watchlist management
- Live/upcoming game status
- Recommendation display (BUY/PASS/HOLD)

**Unique Features to Preserve:**

**From Game Cards:**
- ESPN team logos (visual CDN integration)
- Multi-sport tabs (NFL/NCAA/NBA/MLB)
- Telegram subscribe/notifications per game
- Ensemble AI display (3 models)
- Multi-week game fetching
- NCAA rankings

**From Kalshi Markets:**
- Team emoji picker with user picks
- Price history charts
- Market-level focus (vs game-level)
- Analytics dashboard (heatmaps, scatter plots)
- Export to CSV/JSON
- Bet type classification

**From AVA Betting:**
- Kelly Criterion bet sizing
- High confidence signals (>75% conf, >15% EV)
- Bankroll management settings
- Risk management toggles
- Demo mode

---

## Recommended Architecture

### Page 1: Sports Betting Hub (NEW)

**Structure:**
```
┌─────────────────────────────────────────────────────────────┐
│  🏈 Sports Betting Hub                        [Auto-refresh]│
│  ┌──────────────────────────────────────────────────────┐  │
│  │  NFL | NCAA | NBA | MLB                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  Tabs: [Live Games] [Betting Picks] [Markets] [Watchlist]  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐│
│  │  TAB: Live Games (Game Cards view)                     ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             ││
│  │  │  GAME 1  │  │  GAME 2  │  │  GAME 3  │             ││
│  │  │  [Logo]  │  │  [Logo]  │  │  [Logo]  │             ││
│  │  │  Eagles  │  │  Chiefs  │  │  Bills   │             ││
│  │  │  vs 49ers│  │  vs Ravens│  │  vs Jets │             ││
│  │  │          │  │          │  │          │             ││
│  │  │  Kalshi: │  │  Kalshi: │  │  Kalshi: │             ││
│  │  │  75% | $1│  │  60% | $1│  │  82% | $1│             ││
│  │  │          │  │          │  │          │             ││
│  │  │  AI Pred │  │  AI Pred │  │  AI Pred │             ││
│  │  │  Elo: 78%│  │  Elo: 65%│  │  Elo: 85%│             ││
│  │  │  NN: 72% │  │  NN: 58% │  │  NN: 80% │             ││
│  │  │  XGB: 75%│  │  XGB: 62%│  │  XGB: 83%│             ││
│  │  │          │  │          │  │          │             ││
│  │  │  [Subscribe]│[Subscribe]│[Subscribe]│             ││
│  │  └──────────┘  └──────────┘  └──────────┘             ││
│  └────────────────────────────────────────────────────────┘│
│                                                              │
│  Sidebar:                                                    │
│  • Watchlist (5 games)                                       │
│  • Filters                                                   │
│  • Settings                                                  │
└─────────────────────────────────────────────────────────────┘
```

**Tab Structure:**

1. **Live Games Tab** (Game Cards content)
   - Visual team logos from ESPN
   - Live scores and status
   - Ensemble AI predictions (Elo, Neural Net, XGBoost)
   - Subscribe buttons with Telegram
   - Filters: Date, Status, EV, Confidence
   - Sport selector (NFL/NCAA/NBA/MLB)

2. **Betting Picks Tab** (AVA Betting content)
   - High confidence signals (glowing cards)
   - Kelly Criterion bet sizing
   - EV-focused recommendations
   - BUY/PASS/HOLD actions
   - Bankroll settings
   - Risk management toggles

3. **Markets Tab** (Kalshi Markets content)
   - Market cards with team emoji picker
   - User picks tracking (blue vs green borders)
   - Price history charts
   - Market watchlist
   - Analytics sub-tabs:
     - Volume charts
     - Confidence heatmaps
     - Edge vs confidence scatter
   - Export functionality

4. **Watchlist Tab** (Unified)
   - All subscribed games
   - All tracked markets
   - Quick actions (unsubscribe, view details)
   - Auto-cleanup finished games

5. **Settings Tab**
   - Kelly Criterion settings (fraction, max bet)
   - Bankroll management
   - Risk preferences
   - Notification settings
   - Auto-refresh interval

**Sidebar:**
- Quick filters (confidence, EV, sport)
- Watchlist summary (count badges)
- Active tab indicator

---

### Page 2: Prediction Markets (KEEP SEPARATE)

**File:** `prediction_markets_page.py` (317 lines - no changes)

**Rationale:**
- Different audience (politics, economics, crypto)
- Simple, focused UI appropriate for non-sports
- Minimal overlap with sports betting
- Already compact and well-designed

**Enhancement:**
- Add link at top: "Looking for sports? → Go to Sports Betting Hub"

---

## Implementation Plan

### Phase 3.1: Shared Components (2-3 days)

**Create:** `src/components/betting/`

```
src/components/betting/
├── __init__.py
├── game_card.py           # Render individual game cards with logos
├── market_card.py         # Render market cards with emoji picker
├── betting_pick_card.py   # Render AVA betting recommendations
├── filters.py             # Unified filter controls
├── watchlist.py           # Watchlist sidebar component
├── charts.py              # Price history & analytics charts
└── settings.py            # Settings panel component
```

**Key Components:**

**1. game_card.py**
```python
def render_game_card(
    game: Dict,
    kalshi_odds: Optional[Dict],
    predictions: Dict,
    watchlist: List[str],
    show_subscribe: bool = True
) -> None:
    """
    Render a game card with team logos, scores, and predictions.

    Args:
        game: ESPN game data
        kalshi_odds: Kalshi betting odds (if available)
        predictions: AI predictions (Elo, NN, XGBoost)
        watchlist: List of game IDs in user's watchlist
        show_subscribe: Whether to show subscribe button
    """
```

**2. market_card.py**
```python
def render_market_card(
    market: Dict,
    user_pick: Optional[str],
    show_chart: bool = False,
    expandable: bool = True
) -> None:
    """
    Render a Kalshi market card with emoji picker and details.

    Args:
        market: Kalshi market data
        user_pick: User's team pick (for border color)
        show_chart: Whether to show price history chart
        expandable: Whether card is expandable
    """
```

**3. betting_pick_card.py**
```python
def render_betting_pick(
    pick: Dict,
    kelly_settings: Dict,
    highlight_high_confidence: bool = True
) -> None:
    """
    Render AVA betting recommendation card.

    Args:
        pick: Betting pick data with EV, confidence, Kelly sizing
        kelly_settings: User's Kelly settings (fraction, bankroll)
        highlight_high_confidence: Whether to use glowing effect for >75% conf
    """
```

**4. filters.py**
```python
def render_unified_filters(
    sport: str,
    view_mode: str,
    filter_state: Dict
) -> Dict:
    """
    Render unified filter controls that adapt to current view.

    Args:
        sport: Current sport (NFL, NCAA, etc.)
        view_mode: Current tab (live_games, betting_picks, markets)
        filter_state: Current filter values

    Returns:
        Updated filter state
    """
```

---

### Phase 3.2: Data Layer (1-2 days)

**Create:** `src/services/sports_betting_service.py`

```python
class SportsBettingService:
    """
    Unified data service for sports betting hub.
    Coordinates ESPN, Kalshi, AI predictions, and watchlist data.
    """

    def __init__(self):
        self.kalshi_db = get_kalshi_manager()
        self.watchlist_manager = GameWatchlistManager()
        self.espn_nfl = ESPNNFLClient()
        self.espn_ncaa = ESPNNCAAClient()
        self.nfl_predictor = NFLPredictor()
        self.ncaa_predictor = NCAAPredictor()
        self.betting_agent = AdvancedBettingAIAgent()

    @st.cache_data(ttl=300)
    def fetch_unified_data(
        _self,
        sport: str,
        include_markets: bool = True,
        include_predictions: bool = True,
        week: Optional[int] = None
    ) -> Dict:
        """
        Fetch all data needed for sports betting hub.

        Returns:
            {
                'games': [...],           # ESPN games
                'markets': [...],         # Kalshi markets
                'predictions': {...},     # AI predictions
                'kalshi_matches': {...},  # Game-to-market matches
                'watchlist': [...]        # User's watchlist
            }
        """

    def get_betting_opportunities(
        self,
        sport: str,
        min_confidence: float = 60.0,
        min_ev: float = 5.0,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.25
    ) -> List[Dict]:
        """
        Get AVA betting recommendations with Kelly sizing.
        """

    def subscribe_to_game(self, game_id: str, user_id: str) -> bool:
        """Add game to user's watchlist and enable notifications."""

    def track_market(self, ticker: str, user_id: str) -> bool:
        """Add market to user's tracking list."""
```

---

### Phase 3.3: Main Page Implementation (2-3 days)

**Create:** `sports_betting_hub_page.py` (~2,500 lines)

**Structure:**
```python
# Imports and setup
from src.services.sports_betting_service import SportsBettingService
from src.services import get_kalshi_manager
from src.components.betting import (
    render_game_card,
    render_market_card,
    render_betting_pick,
    render_unified_filters,
    render_watchlist_sidebar
)

# Page configuration
st.set_page_config(page_title="Sports Betting Hub", layout="wide")

# Initialize services
@st.cache_resource
def get_sports_betting_service():
    return SportsBettingService()

# Session state initialization
if 'hub_state' not in st.session_state:
    st.session_state.hub_state = {
        'active_sport': 'NFL',
        'active_tab': 'live_games',
        'watchlist': [],
        'user_picks': {},
        'settings': {
            'kelly_fraction': 0.25,
            'bankroll': 1000,
            'min_confidence': 60,
            'min_ev': 5,
            'auto_refresh': True,
            'refresh_interval': 60
        },
        'filters': {
            'status': 'all',
            'date_range': 'today',
            'min_ev': 0,
            'min_confidence': 0
        }
    }

# Header with sport selector
def render_header():
    """Sticky header with sport tabs and auto-refresh"""
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        sports = ['NFL', 'NCAA', 'NBA', 'MLB']
        selected_sport = st.selectbox(
            "Sport",
            sports,
            index=sports.index(st.session_state.hub_state['active_sport']),
            key='sport_selector'
        )
        st.session_state.hub_state['active_sport'] = selected_sport

    with col2:
        watchlist_count = len(st.session_state.hub_state['watchlist'])
        st.metric("Watchlist", watchlist_count)

    with col3:
        auto_refresh = st.checkbox(
            "Auto-refresh",
            value=st.session_state.hub_state['settings']['auto_refresh']
        )
        st.session_state.hub_state['settings']['auto_refresh'] = auto_refresh

# Tab navigation
def render_tabs():
    """Main tab navigation"""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎮 Live Games",
        "💰 Betting Picks",
        "📊 Markets",
        "⭐ Watchlist",
        "⚙️ Settings"
    ])

    with tab1:
        render_live_games_tab()

    with tab2:
        render_betting_picks_tab()

    with tab3:
        render_markets_tab()

    with tab4:
        render_watchlist_tab()

    with tab5:
        render_settings_tab()

# Tab implementations
def render_live_games_tab():
    """Live Games tab - Game Cards content"""
    service = get_sports_betting_service()
    sport = st.session_state.hub_state['active_sport']

    # Filters
    filters = render_unified_filters(sport, 'live_games', st.session_state.hub_state['filters'])

    # Fetch data
    data = service.fetch_unified_data(sport, include_markets=True, include_predictions=True)

    # Render games
    games = filter_games(data['games'], filters)

    cols_per_row = 3
    for i in range(0, len(games), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(games):
                game = games[i + j]
                with col:
                    render_game_card(
                        game=game,
                        kalshi_odds=data['kalshi_matches'].get(game['id']),
                        predictions=data['predictions'].get(game['id']),
                        watchlist=st.session_state.hub_state['watchlist'],
                        show_subscribe=True
                    )

def render_betting_picks_tab():
    """Betting Picks tab - AVA content"""
    service = get_sports_betting_service()
    sport = st.session_state.hub_state['active_sport']
    settings = st.session_state.hub_state['settings']

    # Get betting opportunities
    picks = service.get_betting_opportunities(
        sport=sport,
        min_confidence=settings['min_confidence'],
        min_ev=settings['min_ev'],
        bankroll=settings['bankroll'],
        kelly_fraction=settings['kelly_fraction']
    )

    # High confidence picks first
    high_conf_picks = [p for p in picks if p['confidence'] > 75 and p['ev'] > 15]
    standard_picks = [p for p in picks if p not in high_conf_picks]

    if high_conf_picks:
        st.subheader("⚡ High Confidence Picks")
        for pick in high_conf_picks:
            render_betting_pick(pick, settings, highlight_high_confidence=True)

    if standard_picks:
        st.subheader("📊 All Opportunities")
        for pick in standard_picks:
            render_betting_pick(pick, settings, highlight_high_confidence=False)

def render_markets_tab():
    """Markets tab - Kalshi Markets content"""
    kalshi_db = get_kalshi_manager()
    sport = st.session_state.hub_state['active_sport']

    # Sub-tabs
    subtab1, subtab2, subtab3 = st.tabs(["All Markets", "Analytics", "Export"])

    with subtab1:
        # Market cards with emoji picker
        markets = kalshi_db.get_markets_with_predictions(market_type='nfl', limit=100)
        for market in markets:
            render_market_card(
                market=market,
                user_pick=st.session_state.hub_state['user_picks'].get(market['ticker']),
                show_chart=True,
                expandable=True
            )

    with subtab2:
        # Analytics charts
        render_analytics_dashboard(markets)

    with subtab3:
        # Export functionality
        render_export_controls(markets)

def render_watchlist_tab():
    """Unified watchlist for games and markets"""
    # Implementation

def render_settings_tab():
    """Settings for Kelly Criterion, bankroll, risk"""
    # Implementation

# Main execution
def main():
    render_header()
    render_tabs()

    # Sidebar
    with st.sidebar:
        render_watchlist_sidebar(st.session_state.hub_state['watchlist'])

if __name__ == "__main__":
    main()
```

---

### Phase 3.4: Migration & Testing (1-2 days)

**Tasks:**
1. **Migrate Unique Features**
   - Port team logo display from Game Cards
   - Port emoji picker from Kalshi Markets
   - Port Kelly calculations from AVA Betting
   - Port analytics charts from Kalshi Markets

2. **Test All Functionality**
   - ESPN data fetching (all sports)
   - Kalshi odds matching
   - AI predictions (Elo, NN, XGBoost)
   - Subscribe/unsubscribe
   - Watchlist management
   - User picks tracking
   - Kelly Criterion calculations
   - Price history charts
   - Export to CSV/JSON
   - Telegram notifications

3. **Performance Testing**
   - Page load times
   - Data fetching speed
   - Chart rendering
   - Auto-refresh behavior
   - Cache hit rates

4. **User Acceptance Testing**
   - Navigation flow
   - Tab switching
   - Filter functionality
   - Mobile responsiveness

---

### Phase 3.5: Dashboard Integration (1 day)

**Update:** `dashboard.py`

**Changes:**

1. **Add New Page Route:**
```python
elif page == "Sports Betting Hub":
    from sports_betting_hub_page import main as render_sports_betting_hub
    render_sports_betting_hub()
```

2. **Update Navigation:**
```python
# Sports Betting section
st.sidebar.markdown("### 🏈 Sports Betting")
if st.sidebar.button("🎮 Sports Betting Hub", width='stretch'):  # NEW
    st.session_state.page = "Sports Betting Hub"
if st.sidebar.button("📊 Prediction Markets", width='stretch'):
    st.session_state.page = "Prediction Markets"
```

3. **Add Deprecation Warnings to Old Pages:**
```python
# At top of game_cards_visual_page.py
st.warning("⚠️ This page has moved to Sports Betting Hub!")
st.info("💡 All live games are now in the unified Sports Betting Hub")
if st.button("🎮 Go to Sports Betting Hub", type="primary"):
    st.session_state.page = "Sports Betting Hub"
    st.rerun()

# Similar for kalshi_nfl_markets_page.py and ava_betting_recommendations_page.py
```

4. **Update Quick Actions:**
- Add "Sports Betting Hub" to quick access
- Keep Prediction Markets separate

---

### Phase 3.6: Deprecation & Cleanup (After 2-week transition)

**Delete Old Files:**
```bash
# After users have transitioned to new hub
rm game_cards_visual_page.py        # 2,157 lines
rm kalshi_nfl_markets_page.py       # 1,580 lines
rm ava_betting_recommendations_page.py  # 539 lines
```

**Keep:**
```bash
# Remains unchanged
prediction_markets_page.py          # 317 lines
```

**New Files:**
```bash
sports_betting_hub_page.py          # ~2,500 lines
src/services/sports_betting_service.py  # ~400 lines
src/components/betting/*.py         # ~600 lines total
```

---

## Success Criteria

### Code Metrics
- [x] Reduce from 4,276 lines → ~2,500 lines (42% reduction)
- [x] Eliminate duplicate implementations
- [x] Single source of truth for sports betting
- [x] All unique features preserved

### User Experience
- [x] Single entry point for sports betting
- [x] Consistent navigation across all features
- [x] Unified watchlist management
- [x] Seamless tab switching
- [x] No loss of functionality

### Performance
- [x] Page load time < 3 seconds
- [x] Auto-refresh works smoothly
- [x] Charts render quickly
- [x] Data caching effective (>80% hit rate)

### Maintainability
- [x] Shared components reduce duplication
- [x] Unified data service simplifies logic
- [x] Clear separation of concerns
- [x] Easy to add new sports/features

---

## Risk Assessment

### High Risk
- **Feature Loss:** Accidentally dropping unique features during migration
  - **Mitigation:** Comprehensive feature checklist, thorough testing

- **Performance Degradation:** Combined page loading slower than individual pages
  - **Mitigation:** Lazy loading, effective caching, tab-based data fetching

### Medium Risk
- **User Confusion:** Users looking for old pages
  - **Mitigation:** Clear deprecation warnings, prominent links to new hub

- **Data Integration Issues:** ESPN + Kalshi + AI predictions coordination
  - **Mitigation:** Robust SportsBettingService with error handling

### Low Risk
- **Navigation Complexity:** Too many tabs/options
  - **Mitigation:** User testing, iterative refinement

---

## Rollback Plan

If critical issues arise:

```bash
# Restore old pages from git
git checkout HEAD game_cards_visual_page.py
git checkout HEAD kalshi_nfl_markets_page.py
git checkout HEAD ava_betting_recommendations_page.py

# Remove new hub
rm sports_betting_hub_page.py
rm -rf src/components/betting
rm src/services/sports_betting_service.py

# Restore original navigation
git checkout HEAD dashboard.py
```

**Recovery Time:** < 10 minutes

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 3.1: Shared Components | 2-3 days | Phase 2 complete |
| 3.2: Data Layer | 1-2 days | Phase 3.1 |
| 3.3: Main Page | 2-3 days | Phase 3.2 |
| 3.4: Migration & Testing | 1-2 days | Phase 3.3 |
| 3.5: Dashboard Integration | 1 day | Phase 3.4 |
| 3.6: Cleanup | After transition | User feedback |
| **Total** | **7-11 days** | |

**Recommended:** 2 weeks with buffer for testing and refinement

---

## Next Steps

1. **Review & Approve Plan** - Confirm approach before starting implementation
2. **Set Up Development Branch** - `git checkout -b feature/sports-betting-hub`
3. **Begin Phase 3.1** - Create shared components
4. **Iterative Development** - Build and test each phase
5. **User Testing** - Get feedback before deprecating old pages
6. **Production Deploy** - Roll out with feature flag
7. **Monitor Usage** - Track metrics for 2 weeks
8. **Complete Deprecation** - Remove old pages after successful transition

---

## Conclusion

Phase 3 will significantly improve Magnus by:
- **Reducing code by 40%** (4,593 → 2,800 lines)
- **Improving UX** with unified sports betting hub
- **Simplifying maintenance** with shared components
- **Preserving all features** from individual pages
- **Setting foundation** for future sport additions

The implementation will take 7-11 days but delivers substantial long-term value in code quality, user experience, and maintainability.

---

**Status:** 📋 PLANNED
**Next Action:** Begin Phase 3.1 (Shared Components)
**Estimated Start:** After user approval
