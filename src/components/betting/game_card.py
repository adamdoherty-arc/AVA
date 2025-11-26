import streamlit as st
from datetime import datetime

def render_game_card(game_data):
    """
    Render a game card with teams, scores, and status
    
    Args:
        game_data (dict): Dictionary containing game information
    """
    with st.container():
        st.markdown(f"""
        <div style="
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        ">
            <div style="display: flex; justify_content: space-between; align_items: center; margin-bottom: 10px;">
                <span style="font-weight: bold; color: #555;">{game_data.get('league', 'NFL')}</span>
                <span style="
                    background-color: {'#e6fffa' if game_data.get('is_live') else '#f7fafc'};
                    color: {'#2c7a7b' if game_data.get('is_live') else '#4a5568'};
                    padding: 2px 8px;
                    border-radius: 4px;
                    font-size: 0.8em;
                    font-weight: 600;
                ">{game_data.get('status', 'Scheduled')}</span>
            </div>
            
            <div style="display: flex; justify_content: space-between; align_items: center;">
                <div style="flex: 1;">
                    <div style="font-size: 1.1em; font-weight: bold;">{game_data.get('away_team')}</div>
                    <div style="color: #718096; font-size: 0.9em;">{game_data.get('away_record', '')}</div>
                </div>
                <div style="font-size: 1.5em; font-weight: bold; padding: 0 15px;">
                    {game_data.get('away_score', '-')}
                </div>
            </div>
            
            <div style="display: flex; justify_content: space-between; align_items: center; margin-top: 10px;">
                <div style="flex: 1;">
                    <div style="font-size: 1.1em; font-weight: bold;">{game_data.get('home_team')}</div>
                    <div style="color: #718096; font-size: 0.9em;">{game_data.get('home_record', '')}</div>
                </div>
                <div style="font-size: 1.5em; font-weight: bold; padding: 0 15px;">
                    {game_data.get('home_score', '-')}
                </div>
            </div>
            
            <div style="margin-top: 10px; font-size: 0.85em; color: #718096; text-align: right;">
                {game_data.get('game_time', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)
