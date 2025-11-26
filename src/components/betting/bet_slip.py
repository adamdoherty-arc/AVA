import streamlit as st

def render_bet_slip():
    """
    Render the betting slip sidebar widget
    """
    st.sidebar.markdown("### 🎫 Bet Slip")
    
    if 'bet_slip' not in st.session_state:
        st.session_state.bet_slip = []
        
    if not st.session_state.bet_slip:
        st.sidebar.info("Your bet slip is empty. Click on odds to add bets.")
    else:
        for i, bet in enumerate(st.session_state.bet_slip):
            with st.sidebar.container():
                st.markdown(f"""
                <div style="
                    background-color: #f7fafc;
                    padding: 10px;
                    border-radius: 5px;
                    border: 1px solid #e2e8f0;
                    margin-bottom: 5px;
                ">
                    <div style="font-weight: bold; font-size: 0.9em;">{bet['team']} {bet['type']}</div>
                    <div style="display: flex; justify_content: space-between; font-size: 0.8em;">
                        <span>{bet['line']}</span>
                        <span>{bet['odds']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        st.sidebar.markdown("---")
        wager = st.sidebar.number_input("Wager Amount ($)", min_value=1.0, value=10.0, step=5.0)
        
        if st.sidebar.button("Place Bet", type="primary", use_container_width=True):
            st.sidebar.success("Bet placed successfully! (Simulation)")
            st.session_state.bet_slip = []
            st.rerun()
            
        if st.sidebar.button("Clear Slip", use_container_width=True):
            st.session_state.bet_slip = []
            st.rerun()
