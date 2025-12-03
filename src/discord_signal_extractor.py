"""
Discord Trading Signal Extractor
Extracts structured trading data from Discord messages for AVA's RAG system
"""
import re
from typing import Dict, List, Optional
from datetime import datetime
import psycopg2
import psycopg2.extras
import os


class DiscordSignalExtractor:
    """Extract structured trading signals from Discord messages"""

    def __init__(self) -> None:
        self.db_password = os.getenv('DB_PASSWORD')

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(
            host='localhost',
            port='5432',
            database='magnus',
            user='postgres',
            password=self.db_password
        )

    def extract_tickers(self, content: str) -> List[str]:
        """Extract stock tickers from message"""
        # Patterns: $TICKER, TICKER (2-5 caps), or common formats
        tickers = set()

        # $TICKER format
        dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', content)
        tickers.update(dollar_tickers)

        # Standalone tickers (2-5 uppercase letters)
        # But filter out common words
        common_words = {'DD', 'CEO', 'FDA', 'IPO', 'ATH', 'ATL', 'BUY', 'SELL', 'CALL', 'PUT'}
        standalone = re.findall(r'\b([A-Z]{2,5})\b', content)
        for ticker in standalone:
            if ticker not in common_words and len(ticker) <= 5:
                tickers.add(ticker)

        return list(tickers)

    def extract_option_info(self, content: str) -> Optional[Dict]:
        """Extract option contract details"""
        option_info = {}

        # Option patterns: "AAPL 150C 12/15", "SPY 450P exp 1/20"
        # Strike + Call/Put
        strike_pattern = r'(\d+(?:\.\d+)?)\s*([CP])\b'
        strike_match = re.search(strike_pattern, content, re.IGNORECASE)

        if strike_match:
            option_info['strike'] = float(strike_match.group(1))
            option_info['type'] = 'CALL' if strike_match.group(2).upper() == 'C' else 'PUT'

        # Expiration patterns
        exp_patterns = [
            r'exp(?:iry)?\s*(\d{1,2}[/-]\d{1,2})',  # exp 12/15
            r'(\d{1,2}[/-]\d{1,2})\s*exp',  # 12/15 exp
            r'DTE\s*(\d+)',  # DTE 7
        ]

        for pattern in exp_patterns:
            exp_match = re.search(pattern, content, re.IGNORECASE)
            if exp_match:
                option_info['expiration'] = exp_match.group(1)
                break

        return option_info if option_info else None

    def extract_prices(self, content: str) -> Dict[str, Optional[float]]:
        """Extract entry, target, and stop prices"""
        prices = {
            'entry': None,
            'target': None,
            'stop_loss': None
        }

        # Entry patterns
        entry_patterns = [
            r'entry[:\s]+\$?(\d+(?:\.\d{2})?)',
            r'buy[:\s]+\$?(\d+(?:\.\d{2})?)',
            r'@\s*\$?(\d+(?:\.\d{2})?)',
        ]
        for pattern in entry_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                prices['entry'] = float(match.group(1))
                break

        # Target patterns
        target_patterns = [
            r'target[:\s]+\$?(\d+(?:\.\d{2})?)',
            r'tp[:\s]+\$?(\d+(?:\.\d{2})?)',
            r'pt[:\s]+\$?(\d+(?:\.\d{2})?)',
        ]
        for pattern in target_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                prices['target'] = float(match.group(1))
                break

        # Stop loss patterns
        stop_patterns = [
            r'stop[:\s]+\$?(\d+(?:\.\d{2})?)',
            r'sl[:\s]+\$?(\d+(?:\.\d{2})?)',
        ]
        for pattern in stop_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                prices['stop_loss'] = float(match.group(1))
                break

        return prices

    def determine_setup_type(self, content: str) -> str:
        """Determine trading setup type"""
        content_lower = content.lower()

        setup_keywords = {
            'earnings_play': ['earnings', 'er play', 'beat', 'miss'],
            'breakout': ['breakout', 'breaking out', 'breach', 'breakthrough'],
            'pullback': ['pullback', 'dip', 'retrace', 'bounce'],
            'reversal': ['reversal', 'reversing', 'bottom', 'top'],
            'momentum': ['momentum', 'strong move', 'trending'],
            'swing': ['swing', 'multiday', 'position'],
            'day_trade': ['day trade', 'scalp', 'quick'],
            'gap_play': ['gap up', 'gap down', 'gapping'],
            'catalyst': ['catalyst', 'news', 'announcement'],
        }

        for setup_type, keywords in setup_keywords.items():
            if any(kw in content_lower for kw in keywords):
                return setup_type

        return 'general'

    def determine_sentiment(self, content: str) -> str:
        """Determine bullish/bearish sentiment"""
        content_lower = content.lower()

        bullish_words = ['bullish', 'long', 'call', 'buy', 'moon', 'rocket', 'breakout', 'strong']
        bearish_words = ['bearish', 'short', 'put', 'sell', 'dump', 'breakdown', 'weak']

        bullish_count = sum(1 for word in bullish_words if word in content_lower)
        bearish_count = sum(1 for word in bearish_words if word in content_lower)

        if bullish_count > bearish_count:
            return 'bullish'
        elif bearish_count > bullish_count:
            return 'bearish'
        else:
            return 'neutral'

    def calculate_confidence(self, signal: Dict) -> int:
        """Calculate confidence score 0-100 based on data completeness"""
        score = 0

        # Has ticker
        if signal.get('tickers'):
            score += 25

        # Has entry price
        if signal.get('entry'):
            score += 20

        # Has target
        if signal.get('target'):
            score += 15

        # Has stop loss
        if signal.get('stop_loss'):
            score += 15

        # Has option info
        if signal.get('option_strike'):
            score += 10

        # Has setup type (not general)
        if signal.get('setup_type') != 'general':
            score += 10

        # Has clear sentiment (not neutral)
        if signal.get('sentiment') != 'neutral':
            score += 5

        return min(100, score)

    def extract_signal(self, message: Dict) -> Optional[Dict]:
        """Extract complete trading signal from a message"""
        content = message.get('content', '')
        if not content or len(content) < 10:
            return None

        # Extract components
        tickers = self.extract_tickers(content)
        if not tickers:
            return None  # No ticker = not a trading signal

        option_info = self.extract_option_info(content)
        prices = self.extract_prices(content)
        setup_type = self.determine_setup_type(content)
        sentiment = self.determine_sentiment(content)

        # Build signal
        signal = {
            'message_id': message.get('message_id'),
            'channel_id': message.get('channel_id'),
            'author': message.get('author_name'),
            'timestamp': message.get('timestamp'),
            'content': content,
            'tickers': tickers,
            'primary_ticker': tickers[0] if tickers else None,
            'setup_type': setup_type,
            'sentiment': sentiment,
            'entry': prices['entry'],
            'target': prices['target'],
            'stop_loss': prices['stop_loss'],
            'option_strike': option_info.get('strike') if option_info else None,
            'option_type': option_info.get('type') if option_info else None,
            'option_expiration': option_info.get('expiration') if option_info else None,
        }

        # Calculate confidence
        signal['confidence'] = self.calculate_confidence(signal)

        # Only return if confidence >= 40% (has at least ticker + one other component)
        return signal if signal['confidence'] >= 40 else None

    def process_all_messages(self) -> int:
        """Process all Discord messages and extract signals"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Get all messages
        cur.execute("""
            SELECT
                m.message_id,
                m.channel_id,
                m.content,
                m.author_name,
                m.timestamp
            FROM discord_messages m
            ORDER BY m.timestamp DESC
        """)
        messages = cur.fetchall()

        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS discord_trading_signals (
                id SERIAL PRIMARY KEY,
                message_id BIGINT REFERENCES discord_messages(message_id) ON DELETE CASCADE,
                channel_id BIGINT,
                author TEXT,
                timestamp TIMESTAMP,
                content TEXT,
                tickers TEXT[],
                primary_ticker TEXT,
                setup_type TEXT,
                sentiment TEXT,
                entry DECIMAL,
                target DECIMAL,
                stop_loss DECIMAL,
                option_strike DECIMAL,
                option_type TEXT,
                option_expiration TEXT,
                confidence INTEGER,
                extracted_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(message_id)
            )
        """)

        # Create index for fast lookups
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_trading_signals_ticker
            ON discord_trading_signals(primary_ticker)
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_trading_signals_timestamp
            ON discord_trading_signals(timestamp DESC)
        """)

        conn.commit()

        # Extract and store signals
        signals_added = 0
        for message in messages:
            signal = self.extract_signal(dict(message))
            if signal:
                try:
                    cur.execute("""
                        INSERT INTO discord_trading_signals (
                            message_id, channel_id, author, timestamp, content,
                            tickers, primary_ticker, setup_type, sentiment,
                            entry, target, stop_loss, option_strike, option_type,
                            option_expiration, confidence
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (message_id) DO UPDATE SET
                            confidence = EXCLUDED.confidence,
                            extracted_at = NOW()
                    """, (
                        signal['message_id'], signal['channel_id'], signal['author'],
                        signal['timestamp'], signal['content'], signal['tickers'],
                        signal['primary_ticker'], signal['setup_type'], signal['sentiment'],
                        signal['entry'], signal['target'], signal['stop_loss'],
                        signal['option_strike'], signal['option_type'],
                        signal['option_expiration'], signal['confidence']
                    ))
                    signals_added += 1
                except Exception as e:
                    print(f"Error storing signal: {e}")
                    continue

        conn.commit()
        cur.close()
        conn.close()

        return signals_added


if __name__ == "__main__":
    extractor = DiscordSignalExtractor()
    count = extractor.process_all_messages()
    print(f"Extracted {count} trading signals from Discord messages")
