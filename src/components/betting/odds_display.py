import streamlit as st

def render_odds_display(odds_data, game_id):
    """
    Render odds buttons for a game
    
    Args:
        odds_data (dict): Dictionary containing odds (spread, moneyline, total)
        game_id (str): Unique game identifier
    """
    
    # Helper to format odds (e.g., -110, +150)
    def fmt_odds(val):
        if val is None: return "-"
        return f"+{val}" if val > 0 else str(val)

    # Helper to create a clickable button-like div (since actual buttons inside loops can be tricky in Streamlit)
    # For now, we'll use Streamlit columns and metrics/buttons
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; font-size: 0.8em; color: #718096; margin-bottom: 2px;">Spread</div>
        """, unsafe_allow_html=True)
        st.button(
            f"{odds_data.get('spread_home', '-')} ({fmt_odds(odds_data.get('spread_home_odds'))})", 
            key=f"spread_{game_id}", 
            use_container_width=True,
            help="Click to add to bet slip"
        )
        
    with col2:
        st.markdown(f"""
        <div style="text-align: center; font-size: 0.8em; color: #718096; margin-bottom: 2px;">Total</div>
        """, unsafe_allow_html=True)
        st.button(
            f"O/U {odds_data.get('total', '-')}", 
            key=f"total_{game_id}", 
            use_container_width=True
        )
        
    with col3:
        st.markdown(f"""
        <div style="text-align: center; font-size: 0.8em; color: #718096; margin-bottom: 2px;">Moneyline</div>
        """, unsafe_allow_html=True)
        st.button(
            f"{fmt_odds(odds_data.get('moneyline_home'))}", 
            key=f"ml_{game_id}", 
            use_container_width=True
        )
