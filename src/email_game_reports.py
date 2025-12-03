"""
Email Game Reports Service
Sends comprehensive game analysis reports via email

Features:
- Daily game summaries with AI predictions
- High-confidence betting opportunities
- Price action alerts
- Customizable HTML formatting
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import List, Dict
import os
from dotenv import load_dotenv
from src.kalshi_db_manager import KalshiDBManager
from src.advanced_betting_ai_agent import AdvancedBettingAIAgent
from src.espn_live_data import get_espn_client
from src.espn_ncaa_live_data import get_espn_ncaa_client
from src.espn_kalshi_matcher_optimized import enrich_games_with_kalshi_odds_optimized as enrich_games_with_kalshi_odds

load_dotenv()
logger = logging.getLogger(__name__)


class EmailGameReportService:
    """Service for generating and emailing game reports"""

    def __init__(self) -> None:
        self.db = KalshiDBManager()
        self.ai_agent = AdvancedBettingAIAgent()
        self.espn_nfl = get_espn_client()
        self.espn_ncaa = get_espn_ncaa_client()

        # Email configuration from .env
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.from_email = os.getenv('EMAIL_FROM', self.smtp_username)
        self.to_email = os.getenv('EMAIL_TO', '')

    def generate_game_report(self, include_all_games: bool = False) -> str:
        """
        Generate comprehensive HTML game report

        Args:
            include_all_games: If True, includes all games. If False, only high-value opportunities

        Returns:
            HTML formatted report
        """

        # Fetch games
        nfl_games = self._fetch_nfl_games()
        ncaa_games = self._fetch_ncaa_games()
        all_games = nfl_games + ncaa_games

        # Analyze with AI
        analyzed_games = []
        for game in all_games:
            try:
                ai_prediction = self.ai_agent.analyze_betting_opportunity(game, {})
                game['ai_prediction'] = ai_prediction
                analyzed_games.append(game)
            except Exception as e:
                logger.warning(f"Could not analyze game {game.get('away_team')} @ {game.get('home_team')}: {e}")

        # Filter by value if requested
        if not include_all_games:
            analyzed_games = [g for g in analyzed_games if g['ai_prediction']['confidence_score'] >= 70]

        # Sort by confidence
        analyzed_games.sort(key=lambda x: x['ai_prediction']['confidence_score'], reverse=True)

        # Generate HTML
        html = self._generate_html_report(analyzed_games, nfl_games, ncaa_games)

        return html

    def _fetch_nfl_games(self) -> List[Dict]:
        """Fetch NFL games from ESPN with Kalshi odds"""
        try:
            games = self.espn_nfl.get_scoreboard()
            # Enrich with Kalshi odds using optimized matcher
            games = enrich_games_with_kalshi_odds(games, self.db)
            return games
        except Exception as e:
            logger.error(f"Error fetching NFL games: {e}")
            return []

    def _fetch_ncaa_games(self) -> List[Dict]:
        """Fetch NCAA games from ESPN with Kalshi odds"""
        try:
            games = self.espn_ncaa.get_scoreboard(group='80')
            # Enrich with Kalshi odds using optimized matcher
            games = enrich_games_with_kalshi_odds(games, self.db)
            return games
        except Exception as e:
            logger.error(f"Error fetching NCAA games: {e}")
            return []

    def _generate_html_report(self, analyzed_games: List[Dict], nfl_games: List[Dict], ncaa_games: List[Dict]) -> str:
        """Generate HTML formatted report"""

        html_parts = []

        # Header
        html_parts.append(f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1 {{ color: #1f2937; border-bottom: 3px solid #3b82f6; padding-bottom: 10px; }}
                h2 {{ color: #374151; margin-top: 30px; }}
                .summary {{ background-color: #eff6ff; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .game-card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; margin: 15px 0; background-color: #fafafa; }}
                .high-confidence {{ border-left: 4px solid #10b981; background-color: #d1fae5; }}
                .medium-confidence {{ border-left: 4px solid #3b82f6; background-color: #dbeafe; }}
                .low-confidence {{ border-left: 4px solid #6b7280; background-color: #f3f4f6; }}
                .game-title {{ font-size: 18px; font-weight: bold; color: #1f2937; margin-bottom: 10px; }}
                .score {{ font-size: 24px; font-weight: bold; color: #3b82f6; }}
                .prediction {{ background-color: white; padding: 15px; border-radius: 6px; margin-top: 10px; }}
                .metric {{ display: inline-block; margin-right: 20px; }}
                .metric-label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; }}
                .metric-value {{ font-size: 18px; font-weight: bold; color: #1f2937; }}
                .reasoning {{ background-color: #f9fafb; padding: 10px; border-radius: 4px; margin-top: 10px; font-size: 14px; }}
                .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 12px; }}
                .lightning {{ color: #f59e0b; font-size: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏈 Daily Sports Betting Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        """)

        # Summary
        high_conf_count = sum(1 for g in analyzed_games if g['ai_prediction']['confidence_score'] >= 75)
        avg_confidence = sum(g['ai_prediction']['confidence_score'] for g in analyzed_games) / len(analyzed_games) if analyzed_games else 0

        html_parts.append(f"""
                <div class="summary">
                    <h2>📊 Today's Summary</h2>
                    <div style="display: flex; justify-content: space-around; margin-top: 15px;">
                        <div style="text-align: center;">
                            <div class="metric-label">Total Games</div>
                            <div class="metric-value">{len(nfl_games) + len(ncaa_games)}</div>
                        </div>
                        <div style="text-align: center;">
                            <div class="metric-label">NFL Games</div>
                            <div class="metric-value">{len(nfl_games)}</div>
                        </div>
                        <div style="text-align: center;">
                            <div class="metric-label">NCAA Games</div>
                            <div class="metric-value">{len(ncaa_games)}</div>
                        </div>
                        <div style="text-align: center;">
                            <div class="metric-label">High-Confidence Bets</div>
                            <div class="metric-value" style="color: #10b981;">{high_conf_count}</div>
                        </div>
                    </div>
                </div>
        """)

        # Games
        if analyzed_games:
            html_parts.append("<h2>🎯 AI-Analyzed Games</h2>")

            for game in analyzed_games[:20]:  # Top 20
                ai_pred = game['ai_prediction']
                confidence = ai_pred['confidence_score']

                # Determine CSS class
                if confidence >= 75:
                    card_class = "game-card high-confidence"
                    conf_label = "⚡ HIGH CONFIDENCE"
                elif confidence >= 60:
                    card_class = "game-card medium-confidence"
                    conf_label = "GOOD OPPORTUNITY"
                else:
                    card_class = "game-card low-confidence"
                    conf_label = "MARGINAL"

                # Winner
                predicted_winner = ai_pred.get('predicted_winner', '').lower()
                if predicted_winner == 'away':
                    winner_name = game.get('away_team', '')
                elif predicted_winner == 'home':
                    winner_name = game.get('home_team', '')
                else:
                    winner_name = 'TBD'

                html_parts.append(f"""
                <div class="{card_class}">
                    <div class="game-title">
                        {game.get('away_team', '')} @ {game.get('home_team', '')}
                    </div>
                    <div class="score">
                        {game.get('away_score', 0)} - {game.get('home_score', 0)}
                    </div>
                    <div style="font-size: 12px; color: #6b7280; margin: 5px 0;">
                        {game.get('status_detail', 'Scheduled')}
                    </div>

                    <div class="prediction">
                        <div style="font-size: 12px; color: #10b981; font-weight: bold; margin-bottom: 8px;">
                            {conf_label}
                        </div>
                        <div style="font-size: 16px; font-weight: bold; margin: 10px 0;">
                            🔼 Predicted Winner: {winner_name}
                        </div>

                        <div style="margin: 15px 0;">
                            <div class="metric">
                                <div class="metric-label">Win Probability</div>
                                <div class="metric-value">{ai_pred.get('win_probability', 0)*100:.0f}%</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Confidence</div>
                                <div class="metric-value">{confidence:.0f}%</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Expected Value</div>
                                <div class="metric-value">{ai_pred.get('expected_value', 0):+.1f}%</div>
                            </div>
                        </div>

                        <div style="background-color: #eff6ff; padding: 8px; border-radius: 4px; margin-top: 10px;">
                            <strong>Recommendation:</strong> {ai_pred.get('recommendation', 'PASS')}
                        </div>

                        <div class="reasoning">
                            <strong>Why:</strong><br>
                            {'<br>'.join(['• ' + r for r in ai_pred.get('reasoning', [])])}
                        </div>
                    </div>
                </div>
                """)
        else:
            html_parts.append("<p>No high-value opportunities found at this time.</p>")

        # Footer
        html_parts.append(f"""
                <div class="footer">
                    <p>This report was generated automatically by the Advanced AI Betting System.</p>
                    <p>All predictions are based on Kelly Criterion and multi-factor analysis.</p>
                    <p><strong>Disclaimer:</strong> This is for informational purposes only. Bet responsibly.</p>
                </div>
            </div>
        </body>
        </html>
        """)

        return ''.join(html_parts)

    def send_email_report(self, subject: str = None, include_all_games: bool = False) -> bool:
        """
        Send email report

        Args:
            subject: Email subject (auto-generated if not provided)
            include_all_games: Include all games or only high-value

        Returns:
            True if sent successfully, False otherwise
        """

        if not self.to_email:
            logger.error("No email recipient configured. Set EMAIL_TO in .env")
            return False

        if not self.smtp_username or not self.smtp_password:
            logger.error("SMTP credentials not configured. Set SMTP_USERNAME and SMTP_PASSWORD in .env")
            return False

        try:
            # Generate report
            logger.info("Generating email report...")
            html_content = self.generate_game_report(include_all_games=include_all_games)

            # Create email
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject or f"Daily Sports Betting Report - {datetime.now().strftime('%B %d, %Y')}"
            msg['From'] = self.from_email
            msg['To'] = self.to_email

            # Attach HTML
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            # Send via SMTP
            logger.info(f"Sending email to {self.to_email}...")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            logger.info("Email sent successfully!")
            return True

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False


def send_daily_report():
    """Convenience function to send daily report"""
    service = EmailGameReportService()
    return service.send_email_report(include_all_games=False)


def send_full_report():
    """Convenience function to send full report with all games"""
    service = EmailGameReportService()
    return service.send_email_report(include_all_games=True)


if __name__ == "__main__":
    # Test sending report
    logging.basicConfig(level=logging.INFO)
    print("Sending daily betting report...")
    success = send_daily_report()
    print(f"Report sent: {success}")
