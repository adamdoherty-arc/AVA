import streamlit as st
from src.services.sports_betting_service import SportsBettingService
from src.components.betting.game_card import render_game_card
from src.components.betting.odds_display import render_odds_display
from src.components.betting.bet_slip import render_bet_slip

def show_sports_betting_hub():
    """
    Render the main Sports Betting Hub page
    """
    st.title("🏟️ Sports Betting Hub")
    
    # Initialize service
    service = SportsBettingService()
    
    # Render Bet Slip in Sidebar
    render_bet_slip()
    
    # Top Stats / Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Live Games", "3")
    with col2:
        st.metric("Open Bets", "5")
    with col3:
        st.metric("ROI (Week)", "+12.5%")
    with col4:
        st.metric("Balance", "$1,250.00")
        
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🔴 Live Action", "📅 Upcoming", "🤖 AI Best Bets"])
    
    with tab1:
        st.subheader("Live Games")
        live_games = service.get_live_games()
        
        if not live_games:
            st.info("No live games currently.")
        else:
            for game in live_games:
                render_game_card(game)
                render_odds_display(game['odds'], game['id'])
                st.markdown("---")
                
    with tab2:
        st.subheader("Upcoming Matchups")
        upcoming_games = service.get_upcoming_games()
        
        if not upcoming_games:
            st.info("No upcoming games found.")
        else:
            for game in upcoming_games:
                render_game_card(game)
                render_odds_display(game['odds'], game['id'])
                st.markdown("---")
                
    with tab3:
        st.subheader("🤖 AI Recommended Bets")
        best_bets = service.get_best_bets()
        
        for bet in best_bets:
            with st.container():
                st.markdown(f"""
                <div style="
                    border-left: 4px solid #48bb78;
                    background-color: #f0fff4;
                    padding: 15px;
                    margin-bottom: 10px;
                    border-radius: 0 5px 5px 0;
                ">
                    <div style="display: flex; justify_content: space-between;">
                        <span style="font-weight: bold; font-size: 1.1em;">{bet['matchup']}</span>
                        <span style="background-color: #48bb78; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em;">{bet['confidence']}% Conf</span>
                    </div>
                    <div style="margin-top: 5px; font-weight: bold; color: #2f855a;">
                        Pick: {bet['pick']} ({bet['odds']})
                    </div>
                    <div style="margin-top: 5px; font-size: 0.9em; color: #555;">
                        {bet['reasoning']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Add to Slip ({bet['pick']})", key=f"ai_bet_{bet['matchup']}"):
                    if 'bet_slip' not in st.session_state:
                        st.session_state.bet_slip = []
                    st.session_state.bet_slip.append({
                        'team': bet['matchup'],
                        'type': 'AI Pick',
                        'line': bet['pick'],
                        'odds': bet['odds']
                    })
                    st.rerun()

if __name__ == "__main__":
    show_sports_betting_hub()
