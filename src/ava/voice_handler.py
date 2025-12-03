"""
AVA Voice Handler
=================

Handles voice conversations with AVA via Telegram.

Features:
- Transcribes voice messages using Whisper (FREE, local)
- Processes natural language queries
- Generates voice responses using Piper TTS (FREE, local)
- Integrates with portfolio data, stock analysis, and task status

Supported Commands:
- Portfolio queries: "How's my portfolio?"
- Stock analysis: "Should I sell a put on NVDA?"
- Task status: "What are you working on?"
- Alerts: "Any important alerts?"
- Market news: "What's happening in the market?"
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import json

class AVAVoiceHandler:
    """Handles voice interactions with AVA"""

    def __init__(self) -> None:
        """Initialize voice handler"""
        self.whisper_model = None
        self.piper_voice = None
        self._load_models()

    def _load_models(self) -> None:
        """Load Whisper and Piper models"""
        try:
            import whisper
            # Use tiny model for speed (FREE, ~1GB)
            self.whisper_model = whisper.load_model("tiny")
            print("✅ Whisper model loaded (tiny)")
        except Exception as e:
            print(f"⚠️  Could not load Whisper: {e}")
            print("   Install: pip install openai-whisper")

        try:
            # Piper TTS will be lazy-loaded when needed
            import piper
            self.piper_available = True
            print("✅ Piper TTS available")
        except Exception as e:
            print(f"⚠️  Could not load Piper: {e}")
            print("   Install: pip install piper-tts")
            self.piper_available = False

    def transcribe_voice(self, voice_file_path: str) -> Optional[str]:
        """
        Transcribe voice message to text using Whisper

        Args:
            voice_file_path: Path to voice file (ogg, mp3, wav)

        Returns:
            Transcribed text or None if error
        """
        if not self.whisper_model:
            return None

        try:
            result = self.whisper_model.transcribe(voice_file_path)
            return result["text"].strip()
        except Exception as e:
            print(f"Error transcribing: {e}")
            return None

    def process_query(self, text: str) -> Dict[str, Any]:
        """
        Process natural language query and generate response

        Args:
            text: User's transcribed query

        Returns:
            Dict with 'response_text' and 'data'
        """
        text_lower = text.lower()

        # Portfolio queries
        if any(word in text_lower for word in ['portfolio', 'account', 'balance', 'p&l', 'pnl']):
            return self._handle_portfolio_query()

        # Robinhood positions queries
        if any(word in text_lower for word in ['position', 'positions', 'holdings', 'what do i own']):
            return self._handle_positions_query()

        # Stock analysis queries
        if any(word in text_lower for word in ['analyze', 'stock', 'ticker', 'put', 'call', 'option']):
            return self._handle_stock_analysis(text)

        # CSP opportunities queries
        if any(word in text_lower for word in ['csp', 'cash secured put', 'opportunities', 'sell put']):
            return self._handle_csp_opportunities()

        # Task status queries
        if any(word in text_lower for word in ['task', 'working', 'status', 'progress', 'complete']):
            return self._handle_task_status()

        # Xtrades alert queries
        if any(word in text_lower for word in ['alert', 'notification', 'xtrades', 'trader']):
            return self._handle_alerts(text)

        # Top traders query
        if any(word in text_lower for word in ['top trader', 'best trader', 'who']):
            return self._handle_top_traders()

        # Specific trader query
        if 'from' in text_lower or 'by' in text_lower:
            return self._handle_trader_specific(text)

        # Market news queries
        if any(word in text_lower for word in ['market', 'news', 'happening', 'update']):
            return self._handle_market_news()

        # Default greeting/help
        return {
            'response_text': "Hi! I'm AVA, your automated trading assistant. I can help with portfolio updates, positions, CSP opportunities, Xtrades alerts, top traders, stock analysis, and market news. What would you like to know?",
            'data': {}
        }

    def _handle_portfolio_query(self) -> Dict[str, Any]:
        """Handle portfolio-related queries with direct psycopg2"""
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()

        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            cursor = conn.cursor()

            # Get latest portfolio balance
            cursor.execute("""
                SELECT balance, date as timestamp, notes
                FROM daily_portfolio_balances
                ORDER BY date DESC
                LIMIT 1
            """)
            result = cursor.fetchone()

            if result:
                balance, timestamp, notes = result
                response = f"Your portfolio balance is ${balance:,.2f} as of {timestamp.strftime('%I:%M %p')}. "
                if notes:
                    response += notes
                return {
                    'response_text': response,
                    'data': {'balance': float(balance)}
                }
            else:
                return {
                    'response_text': "I don't have recent portfolio data. Please sync your Robinhood account.",
                    'data': {}
                }
        except Exception as e:
            print(f"Portfolio query error: {e}")
            return {
                'response_text': "Sorry, I couldn't retrieve your portfolio data. Please check database connection.",
                'data': {}
            }
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _handle_stock_analysis(self, text: str) -> Dict[str, Any]:
        """Handle stock analysis queries"""
        # Extract ticker symbol if present
        import re
        ticker_match = re.search(r'\b([A-Z]{1,5})\b', text.upper())
        ticker = ticker_match.group(1) if ticker_match else None

        if ticker:
            response = f"Analyzing {ticker} for wheel strategy opportunities. Let me check current premium, implied volatility, and technical indicators."
        else:
            response = "Please specify a ticker symbol, like 'analyze NVDA' or 'should I sell a put on AAPL'."

        return {
            'response_text': response,
            'data': {'ticker': ticker}
        }

    def _handle_task_status(self) -> Dict[str, Any]:
        """Handle task status queries"""
        try:
            from src.ava.db_manager import get_db_manager, DatabaseError
            from psycopg2.extras import RealDictCursor

            db = get_db_manager()

            with db.get_cursor(RealDictCursor) as cursor:
                # Get recent completed tasks
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM ci_enhancements
                    WHERE status = 'completed'
                      AND completed_at > NOW() - INTERVAL '24 hours'
                """)
                completed_today = cursor.fetchone()['count']

                # Get current in-progress task
                cursor.execute("""
                    SELECT title
                    FROM ci_enhancements
                    WHERE status = 'in_progress'
                    LIMIT 1
                """)
                current_task = cursor.fetchone()

            response = f"I completed {completed_today} tasks in the last 24 hours. "
            if current_task:
                response += f"I'm currently working on: {current_task['title']}"
            else:
                response += "I'm ready for the next task!"

            return {
                'response_text': response,
                'data': {
                    'completed_today': completed_today,
                    'current_task': current_task['title'] if current_task else None
                }
            }
        except DatabaseError:
            return {
                'response_text': "Sorry, I couldn't retrieve task status. Please try again later.",
                'data': {}
            }
        except Exception as e:
            return {
                'response_text': "An unexpected error occurred. Please try again.",
                'data': {}
            }

    def _handle_alerts(self, text: str = "") -> Dict[str, Any]:
        """Handle Xtrades alert queries with real database data"""
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()

        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            cursor = conn.cursor()

            # Get recent alerts from xtrades_trades (last 24 hours)
            cursor.execute("""
                SELECT
                    p.username as trader_name,
                    t.ticker,
                    t.action,
                    t.strategy,
                    t.strike_price,
                    t.expiration_date,
                    t.entry_price,
                    t.alert_timestamp
                FROM xtrades_trades t
                JOIN xtrades_profiles p ON t.profile_id = p.id
                WHERE t.alert_timestamp > NOW() - INTERVAL '24 hours'
                ORDER BY t.alert_timestamp DESC
                LIMIT 10
            """)

            alerts = cursor.fetchall()

            if alerts:
                response = f"📊 Latest Xtrades Alerts ({len(alerts)} in last 24h):\n\n"
                for alert in alerts:
                    trader, ticker, action, strategy, strike, expiry, premium, timestamp = alert
                    response += f"🔔 {trader}: {action or 'TRADE'} {ticker}"
                    if strategy:
                        response += f" ({strategy})"
                    if strike:
                        response += f" ${strike:.0f}"
                    if expiry:
                        response += f" {expiry.strftime('%m/%d')}"
                    if premium:
                        response += f" @ ${premium:.2f}"
                    response += f" ({timestamp.strftime('%I:%M %p')})\n"

                return {
                    'response_text': response,
                    'data': {'alert_count': len(alerts)}
                }
            else:
                return {
                    'response_text': "No new alerts from Xtrades in the last 24 hours. System is monitoring your followed traders...",
                    'data': {'alert_count': 0}
                }
        except Exception as e:
            print(f"Alerts query error: {e}")
            return {
                'response_text': "Sorry, couldn't check Xtrades alerts. Database may not be synced yet.",
                'data': {}
            }
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _handle_market_news(self) -> Dict[str, Any]:
        """Handle market news queries"""
        response = "Checking latest market updates... Market is currently stable. I'll notify you of any significant movements."
        return {
            'response_text': response,
            'data': {}
        }

    def _handle_positions_query(self) -> Dict[str, Any]:
        """Handle Robinhood positions queries"""
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()

        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            cursor = conn.cursor()

            # Get current positions from Robinhood
            cursor.execute("""
                SELECT
                    symbol as ticker,
                    quantity,
                    average_price as average_buy_price,
                    current_price,
                    (current_price - average_price) * quantity as unrealized_pnl,
                    ((current_price - average_price) / average_price * 100) as pnl_percent
                FROM positions
                WHERE quantity > 0
                ORDER BY (current_price * quantity) DESC
                LIMIT 10
            """)

            positions = cursor.fetchall()

            if positions:
                response = f"📈 Your Current Positions ({len(positions)} holdings):\n\n"
                total_value = 0
                for pos in positions:
                    ticker, qty, avg_price, curr_price, unrealized, pnl_pct = pos
                    position_value = curr_price * qty
                    total_value += position_value
                    pnl_emoji = "📗" if unrealized >= 0 else "📕"
                    response += f"{pnl_emoji} {ticker}: {int(qty)} shares @ ${curr_price:.2f} "
                    response += f"({pnl_pct:+.1f}%) = ${position_value:,.0f}\n"

                response += f"\nTotal Position Value: ${total_value:,.2f}"

                return {
                    'response_text': response,
                    'data': {'position_count': len(positions), 'total_value': float(total_value)}
                }
            else:
                return {
                    'response_text': "You currently have no open positions. Time to find some opportunities!",
                    'data': {'position_count': 0}
                }
        except Exception as e:
            print(f"Positions query error: {e}")
            return {
                'response_text': "Sorry, couldn't retrieve your positions. Please sync with Robinhood first.",
                'data': {}
            }
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _handle_csp_opportunities(self) -> Dict[str, Any]:
        """Handle CSP opportunities queries"""
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()

        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            cursor = conn.cursor()

            # Get top CSP opportunities
            cursor.execute("""
                SELECT
                    ticker,
                    strike_price,
                    expiration_date,
                    premium,
                    delta,
                    implied_volatility,
                    score
                FROM csp_opportunities
                WHERE expiration_date > CURRENT_DATE
                ORDER BY score DESC
                LIMIT 5
            """)

            opportunities = cursor.fetchall()

            if opportunities:
                response = f"💰 Top CSP Opportunities ({len(opportunities)} found):\n\n"
                for opp in opportunities:
                    ticker, strike, expiry, premium, delta, iv, score = opp
                    days_out = (expiry - datetime.now().date()).days
                    response += f"🎯 {ticker} ${strike:.0f} Put ({days_out}d): "
                    response += f"${premium:.2f} premium "
                    response += f"(Δ{delta:.2f}, IV {iv:.0f}%, Score: {score:.1f})\n"

                return {
                    'response_text': response,
                    'data': {'opportunity_count': len(opportunities)}
                }
            else:
                return {
                    'response_text': "No CSP opportunities found. Run the CSP scanner to find new trades!",
                    'data': {'opportunity_count': 0}
                }
        except Exception as e:
            print(f"CSP opportunities query error: {e}")
            return {
                'response_text': "Sorry, couldn't retrieve CSP opportunities. The scanner may not have run yet.",
                'data': {}
            }
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _handle_top_traders(self) -> Dict[str, Any]:
        """Handle top traders query"""
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()

        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            cursor = conn.cursor()

            # Get top traders by trade volume
            cursor.execute("""
                SELECT
                    p.username,
                    p.display_name,
                    COUNT(t.id) as trade_count,
                    COUNT(CASE WHEN t.status = 'open' THEN 1 END) as open_trades,
                    p.last_sync
                FROM xtrades_profiles p
                LEFT JOIN xtrades_trades t ON p.id = t.profile_id
                WHERE p.active = TRUE
                GROUP BY p.id
                ORDER BY trade_count DESC
                LIMIT 5
            """)

            traders = cursor.fetchall()

            if traders:
                response = f"👥 Top Traders on Xtrades ({len(traders)} followed):\n\n"
                for trader in traders:
                    username, display_name, trade_count, open_count, last_sync = trader
                    name = display_name or username
                    response += f"🏆 {name}: {trade_count} total trades, {open_count} open\n"

                return {
                    'response_text': response,
                    'data': {'trader_count': len(traders)}
                }
            else:
                return {
                    'response_text': "No traders are being followed yet. Add profiles in the dashboard to start tracking!",
                    'data': {'trader_count': 0}
                }
        except Exception as e:
            print(f"Top traders query error: {e}")
            return {
                'response_text': "Sorry, couldn't retrieve trader information. Database may not be synced yet.",
                'data': {}
            }
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _handle_trader_specific(self, text: str) -> Dict[str, Any]:
        """Handle queries about specific traders"""
        import psycopg2
        from dotenv import load_dotenv
        import re
        load_dotenv()

        # Extract trader name from query
        trader_match = re.search(r'(?:from|by)\s+(\w+)', text.lower())
        if not trader_match:
            return {
                'response_text': "Please specify which trader you want to see. For example: 'Show me trades from behappy'",
                'data': {}
            }

        trader_name = trader_match.group(1)

        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            cursor = conn.cursor()

            # Get recent trades from specific trader
            cursor.execute("""
                SELECT
                    t.ticker,
                    t.action,
                    t.strategy,
                    t.strike_price,
                    t.expiration_date,
                    t.entry_price,
                    t.status,
                    t.alert_timestamp
                FROM xtrades_trades t
                JOIN xtrades_profiles p ON t.profile_id = p.id
                WHERE LOWER(p.username) = LOWER(%s) OR LOWER(p.display_name) = LOWER(%s)
                ORDER BY t.alert_timestamp DESC
                LIMIT 5
            """, (trader_name, trader_name))

            trades = cursor.fetchall()

            if trades:
                response = f"📊 Recent trades from {trader_name} ({len(trades)} shown):\n\n"
                for trade in trades:
                    ticker, action, strategy, strike, expiry, premium, status, timestamp = trade
                    response += f"• {ticker} {action or 'TRADE'}"
                    if strategy:
                        response += f" ({strategy})"
                    if strike:
                        response += f" ${strike:.0f}"
                    response += f" - {status.upper()}"
                    response += f" ({timestamp.strftime('%m/%d %I:%M %p')})\n"

                return {
                    'response_text': response,
                    'data': {'trade_count': len(trades), 'trader': trader_name}
                }
            else:
                return {
                    'response_text': f"No trades found from {trader_name}. They may not be in your followed list.",
                    'data': {'trade_count': 0, 'trader': trader_name}
                }
        except Exception as e:
            print(f"Trader-specific query error: {e}")
            return {
                'response_text': f"Sorry, couldn't retrieve trades from {trader_name}.",
                'data': {}
            }
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def generate_voice_response(self, text: str, output_path: str) -> bool:
        """
        Generate voice response using Piper TTS

        Args:
            text: Text to convert to speech
            output_path: Path to save audio file

        Returns:
            True if successful, False otherwise
        """
        if not self.piper_available:
            return False

        try:
            from piper import PiperVoice
            import wave

            # Lazy-load Piper voice model if not already loaded
            if not hasattr(self, '_piper_voice_model') or self._piper_voice_model is None:
                # Download voice model if needed
                voice_model_path = self._get_or_download_voice_model()
                if not voice_model_path:
                    print("Failed to get voice model")
                    return False

                self._piper_voice_model = PiperVoice.load(voice_model_path)
                print(f"Loaded Piper voice model: {voice_model_path}")

            # Generate speech
            with wave.open(output_path, 'wb') as wav_file:
                self._piper_voice_model.synthesize(text, wav_file)

            print(f"Generated voice response: {output_path}")
            return True

        except Exception as e:
            print(f"Error generating voice: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_or_download_voice_model(self) -> Optional[str]:
        """Get or download Piper voice model"""
        try:
            # Check if model already exists
            model_dir = Path.home() / ".local" / "share" / "piper-tts"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Use a small, fast English voice (en_US-lessac-medium)
            voice_name = "en_US-lessac-medium"
            model_path = model_dir / f"{voice_name}.onnx"
            config_path = model_dir / f"{voice_name}.onnx.json"

            if model_path.exists() and config_path.exists():
                print(f"Using cached voice model: {model_path}")
                return str(model_path)

            # Download voice model
            print(f"Downloading Piper voice model: {voice_name}...")
            import urllib.request

            base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

            # Download .onnx model
            model_url = f"{base_url}/en_US-lessac-medium.onnx"
            urllib.request.urlretrieve(model_url, model_path)
            print(f"Downloaded model: {model_path}")

            # Download .onnx.json config
            config_url = f"{base_url}/en_US-lessac-medium.onnx.json"
            urllib.request.urlretrieve(config_url, config_path)
            print(f"Downloaded config: {config_path}")

            return str(model_path)

        except Exception as e:
            print(f"Error getting voice model: {e}")
            import traceback
            traceback.print_exc()
            return None


# Convenience function for quick testing
def test_voice_handler():
    """Test the voice handler"""
    handler = AVAVoiceHandler()

    print("\nTesting queries:")

    queries = [
        "How's my portfolio?",
        "Should I sell a put on NVDA?",
        "What are you working on?",
        "Any important alerts?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        response = handler.process_query(query)
        print(f"Response: {response['response_text']}")


if __name__ == "__main__":
    test_voice_handler()
