"""
Sector Classifier - Classify stocks by GICS sector
Simplified version for rapid deployment
"""

import os
import psycopg2
import yfinance as yf
from typing import List, Dict
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SECTOR_MAPPING = {
    'Technology': 'Technology',
    'Healthcare': 'Healthcare',
    'Financial Services': 'Financials',
    'Financials': 'Financials',
    'Consumer Cyclical': 'Consumer Discretionary',
    'Consumer Defensive': 'Consumer Staples',
    'Industrials': 'Industrials',
    'Basic Materials': 'Materials',
    'Energy': 'Energy',
    'Utilities': 'Utilities',
    'Real Estate': 'Real Estate',
    'Communication Services': 'Communication Services',
}

class SectorClassifier:
    def __init__(self) -> None:
        self.conn = None

    def connect(self) -> None:
        self.conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres123!'),
            database=os.getenv('DB_NAME', 'magnus')
        )

    def get_unclassified_stocks(self, limit=100):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT sd.symbol
            FROM stock_data sd
            LEFT JOIN stock_sectors ss ON sd.symbol = ss.symbol
            WHERE ss.symbol IS NULL
            LIMIT %s
        """, (limit,))
        symbols = [row[0] for row in cur.fetchall()]
        cur.close()
        return symbols

    def classify_stock(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                return None

            raw_sector = info.get('sector', 'Unknown')
            sector = SECTOR_MAPPING.get(raw_sector, raw_sector)
            industry = info.get('industry', 'Unknown')
            market_cap = info.get('marketCap', 0)

            return {
                'symbol': symbol,
                'sector': sector,
                'industry': industry,
                'market_cap': market_cap
            }
        except Exception as e:
            logger.error(f"Error classifying {symbol}: {e}")
            return None

    def save_classification(self, data):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO stock_sectors (symbol, sector, industry, market_cap)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (symbol) DO UPDATE SET
                sector = EXCLUDED.sector,
                industry = EXCLUDED.industry,
                market_cap = EXCLUDED.market_cap,
                last_updated = NOW()
        """, (data['symbol'], data['sector'], data['industry'], data['market_cap']))
        self.conn.commit()
        cur.close()

    def classify_batch(self, limit=100):
        self.connect()
        symbols = self.get_unclassified_stocks(limit)

        logger.info(f"Classifying {len(symbols)} stocks...")
        success = 0

        for symbol in symbols:
            data = self.classify_stock(symbol)
            if data:
                self.save_classification(data)
                success += 1
            time.sleep(0.5)  # Rate limiting

        logger.info(f"Classified {success}/{len(symbols)} stocks")
        self.conn.close()

        return success

if __name__ == "__main__":
    classifier = SectorClassifier()
    classifier.classify_batch(100)
