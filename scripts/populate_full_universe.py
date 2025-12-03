"""
Populate FULL Stocks and ETFs Universe
Comprehensive coverage: S&P 500, Russell 1000, Nasdaq 100, and all major ETFs
"""

import os
import psycopg2
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Set
import logging
from dotenv import load_dotenv
import time
import requests

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_value(val):
    """Convert numpy types to Python native types"""
    if val is None:
        return None
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    return val


class FullUniversePopulator:
    """Populate complete stocks and ETFs universe"""

    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
            'database': os.getenv('DB_NAME', 'magnus')
        }
        self.conn = psycopg2.connect(**self.db_config)
        self.processed_stocks: Set[str] = set()
        self.processed_etfs: Set[str] = set()

    def get_existing_symbols(self):
        """Get already processed symbols from database"""
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT symbol FROM stocks_universe")
            self.processed_stocks = {row[0] for row in cur.fetchall()}
            cur.execute("SELECT symbol FROM etfs_universe")
            self.processed_etfs = {row[0] for row in cur.fetchall()}
            logger.info(f"Already have {len(self.processed_stocks)} stocks, {len(self.processed_etfs)} ETFs")
        except Exception:
            pass
        finally:
            cur.close()

    def get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 components"""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers)
            tables = pd.read_html(response.text)
            df = tables[0]
            symbols = df['Symbol'].str.replace('.', '-', regex=False).tolist()
            logger.info(f"Found {len(symbols)} S&P 500 stocks")
            return symbols
        except Exception as e:
            logger.error(f"Error fetching S&P 500: {e}")
            # Fallback: hardcoded list of major S&P 500 stocks
            return self._get_sp500_fallback()

    def _get_sp500_fallback(self) -> List[str]:
        """Fallback list of major S&P 500 stocks"""
        return [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM',
            'ORCL', 'ACN', 'IBM', 'INTC', 'AMD', 'TXN', 'QCOM', 'NOW', 'AMAT', 'ADI',
            'INTU', 'SNPS', 'CDNS', 'KLAC', 'LRCX', 'MCHP', 'MU', 'FTNT', 'PANW', 'CRWD',
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'SPGI',
            'CME', 'ICE', 'MCO', 'CB', 'PGR', 'AON', 'MMC', 'TRV', 'MET', 'AIG',
            'AFL', 'PRU', 'ALL', 'USB', 'PNC', 'TFC', 'BK', 'STT', 'COF', 'DFS',
            # Healthcare
            'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
            'AMGN', 'MDT', 'GILD', 'CVS', 'CI', 'ISRG', 'ELV', 'VRTX', 'REGN', 'SYK',
            'BSX', 'ZTS', 'BDX', 'HUM', 'MRNA', 'BIIB', 'IQV', 'DXCM', 'EW', 'A',
            # Consumer
            'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST', 'TJX',
            'WMT', 'PG', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'EL', 'KMB',
            'GIS', 'K', 'HSY', 'STZ', 'SJM', 'CPB', 'HRL', 'CHD', 'CLX', 'CAG',
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY',
            'KMI', 'WMB', 'HAL', 'DVN', 'BKR', 'FANG', 'HES', 'TRGP', 'OKE', 'CTRA',
            # Industrials
            'CAT', 'DE', 'HON', 'UNP', 'RTX', 'BA', 'GE', 'LMT', 'MMM', 'UPS',
            'NOC', 'GD', 'CSX', 'NSC', 'WM', 'EMR', 'ITW', 'ETN', 'FDX', 'PH',
            'JCI', 'TDG', 'CMI', 'PCAR', 'ROK', 'CARR', 'OTIS', 'IR', 'DOV', 'AME',
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'XEL', 'PCG', 'EXC', 'ED',
            'WEC', 'ES', 'AWK', 'DTE', 'FE', 'PPL', 'CEG', 'EIX', 'AES', 'ETR',
            # Real Estate
            'PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
            'EQR', 'VTR', 'IRM', 'SBAC', 'ARE', 'EXR', 'MAA', 'ESS', 'UDR', 'CBRE',
            # Materials
            'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NUE', 'VMC', 'NEM', 'MLM', 'DOW',
            'DD', 'ALB', 'PPG', 'CTVA', 'CE', 'LYB', 'IFF', 'FMC', 'EMN', 'AMCR',
            # Communication
            'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'EA', 'WBD', 'OMC',
            'TTWO', 'IPG', 'MTCH', 'FOXA', 'FOX', 'PARA', 'NWS', 'NWSA', 'LYV', 'DISH',
            # More Tech
            'PYPL', 'SHOP', 'SQ', 'SNOW', 'DDOG', 'NET', 'ZS', 'OKTA', 'TEAM', 'TWLO',
            'DOCU', 'SPLK', 'WDAY', 'ZM', 'VEEV', 'COUP', 'HUBS', 'TTD', 'BILL', 'CFLT',
            'MDB', 'PATH', 'FIVN', 'PTC', 'ANSS', 'MANH', 'TYL', 'CDAY', 'PAYC', 'CPRT',
            # Retail & Consumer Services
            'ABNB', 'BKNG', 'MAR', 'HLT', 'H', 'RCL', 'CCL', 'NCLH', 'WYNN', 'MGM',
            'LVS', 'EXPE', 'ORLY', 'AZO', 'AAP', 'BBY', 'GME', 'ULTA', 'DG', 'DLTR',
            'ROST', 'GPS', 'KSS', 'M', 'JWN', 'FL', 'ANF', 'URBN', 'VFC', 'PVH',
            # Financial Services
            'V', 'MA', 'FIS', 'FISV', 'GPN', 'NDAQ', 'MSCI', 'FLT', 'WU', 'ALLY',
            'SYF', 'CFG', 'FITB', 'KEY', 'RF', 'HBAN', 'ZION', 'CMA', 'SIVB', 'SBNY',
            # Healthcare Equipment & Services
            'BAX', 'CAH', 'MCK', 'ABC', 'COR', 'HOLX', 'ALGN', 'TFX', 'HSIC', 'WST',
            'WAT', 'PKI', 'TECH', 'MTD', 'DGX', 'LH', 'CRL', 'RMD', 'HCA', 'UHS',
            # Semiconductors
            'NXPI', 'ON', 'SWKS', 'MPWR', 'QRVO', 'MRVL', 'GFS', 'ENTG', 'TER', 'CRUS',
            'WOLF', 'SLAB', 'LSCC', 'ALGM', 'FORM', 'RMBS', 'POWI', 'IPGP', 'COHR', 'MKSI',
            # Auto & Transport
            'GM', 'F', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'APTV', 'BWA', 'ALV',
            'LEA', 'MGA', 'VC', 'GNTX', 'DAN', 'THRM', 'AXL', 'CWH', 'LKQ', 'MOD',
            # Aerospace & Defense
            'TDG', 'HWM', 'HII', 'LHX', 'LDOS', 'BAH', 'CACI', 'SAIC', 'MRCY', 'KTOS',
            'AJRD', 'AVAV', 'TXT', 'SPR', 'TGI', 'VSAT', 'HEICO', 'HEI', 'AXON', 'OSK',
            # Food & Beverage
            'MNST', 'BF-B', 'TAP', 'SAM', 'COKE', 'KDP', 'BG', 'ADM', 'TSN', 'HRL',
            'SYY', 'USFD', 'PFGC', 'DAR', 'FDP', 'PPC', 'CALM', 'JJSF', 'INGR', 'LNDC',
            # Biotech
            'ILMN', 'RGEN', 'SGEN', 'EXAS', 'ALNY', 'SRPT', 'BMRN', 'INCY', 'IONS', 'UTHR',
            'NBIX', 'HZNP', 'EXEL', 'RARE', 'BLUE', 'SAGE', 'ACAD', 'FOLD', 'ARNA', 'ARWR',
            # Insurance
            'HIG', 'L', 'LNC', 'GL', 'KMPR', 'ORI', 'SIGI', 'ACGL', 'RNR', 'CINF',
            'WRB', 'AIZ', 'EG', 'AFG', 'PFG', 'VOYA', 'UNUM', 'RGA', 'CNO', 'BHF'
        ]

    def get_nasdaq100_symbols(self) -> List[str]:
        """Get Nasdaq 100 components"""
        try:
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers)
            tables = pd.read_html(response.text)
            for table in tables:
                if 'Ticker' in table.columns:
                    symbols = table['Ticker'].tolist()
                    logger.info(f"Found {len(symbols)} Nasdaq 100 stocks")
                    return symbols
                if 'Symbol' in table.columns:
                    symbols = table['Symbol'].tolist()
                    logger.info(f"Found {len(symbols)} Nasdaq 100 stocks")
                    return symbols
            return []
        except Exception as e:
            logger.error(f"Error fetching Nasdaq 100: {e}")
            return []

    def get_dow30_symbols(self) -> List[str]:
        """Get Dow Jones 30 components"""
        return [
            'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX',
            'DIS', 'DOW', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ',
            'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG',
            'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
        ]

    def get_russell1000_symbols(self) -> List[str]:
        """Get Russell 1000 components (approximation via popular stocks)"""
        # Major stocks not in S&P 500 but in Russell 1000
        additional = [
            # Tech
            'DDOG', 'NET', 'CRWD', 'ZS', 'SNOW', 'PLTR', 'U', 'RBLX',
            'COIN', 'HOOD', 'AFRM', 'UPST', 'SOFI', 'BILL', 'HUBS',
            'TWLO', 'ZM', 'DOCU', 'ROKU', 'TTD', 'PINS', 'SNAP',
            # Biotech
            'MRNA', 'BNTX', 'SGEN', 'ALNY', 'EXAS', 'INCY', 'NBIX',
            # Energy
            'OXY', 'DVN', 'FANG', 'MRO', 'APA', 'EQT', 'AR', 'RRC',
            # Consumer
            'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'FSR',
            # Finance
            'SQ', 'PYPL', 'MELI', 'SE', 'SHOP', 'WIX', 'BIGC',
            # Healthcare
            'TDOC', 'HIMS', 'DOCS', 'AMWL', 'TALK', 'GDRX',
            # Industrial
            'PLUG', 'FCEL', 'BE', 'ENPH', 'SEDG', 'RUN', 'NOVA',
            # Crypto/Fintech
            'MSTR', 'MARA', 'RIOT', 'CLSK', 'HUT', 'BTBT', 'CIFR',
            # Space/Defense
            'RKLB', 'SPCE', 'ASTS', 'RDW', 'BKSY', 'PL',
            # AI/Semiconductors
            'ARM', 'SMCI', 'IONQ', 'RGTI', 'QUBT',
            # Gaming/Entertainment
            'RBLX', 'DKNG', 'PENN', 'CHDN', 'MGM', 'WYNN', 'LVS',
            # Retail
            'CHWY', 'ETSY', 'W', 'CVNA', 'CARG', 'REAL',
            # Food/Bev
            'CELH', 'MNST', 'SAM', 'FIZZ',
            # Travel
            'ABNB', 'EXPE', 'BKNG', 'TRIP', 'TCOM',
            # More Tech
            'PATH', 'MDB', 'ESTC', 'CFLT', 'DBX', 'BOX', 'PD',
            'SMAR', 'FROG', 'ASAN', 'TEAM', 'ATLCY',
        ]
        return additional

    def get_comprehensive_etf_list(self) -> List[str]:
        """Get comprehensive list of all major ETFs"""
        etfs = [
            # Major Index ETFs
            'SPY', 'IVV', 'VOO', 'VTI', 'QQQ', 'DIA', 'IWM', 'IWB',
            'IWV', 'VTV', 'VUG', 'MTUM', 'QUAL', 'VLUE', 'SIZE',
            'RSP', 'SPLG', 'SPTM', 'ITOT', 'SCHB', 'SCHX', 'VV',

            # Sector ETFs - Technology
            'XLK', 'VGT', 'FTEC', 'IGV', 'SOXX', 'SMH', 'XSD',
            'ARKK', 'ARKW', 'ARKG', 'ARKF', 'ARKQ', 'ARKX',
            'WCLD', 'CLOU', 'SKYY', 'HACK', 'BUG', 'CIBR',
            'BOTZ', 'ROBO', 'IRBO', 'AIEQ', 'AIQ', 'CHAT',

            # Sector ETFs - Financial
            'XLF', 'VFH', 'IYF', 'KBE', 'KRE', 'IAI', 'IYG',

            # Sector ETFs - Healthcare
            'XLV', 'VHT', 'IBB', 'XBI', 'IHI', 'ARKG', 'GNOM',
            'IDNA', 'LABU', 'LABD',

            # Sector ETFs - Energy
            'XLE', 'VDE', 'XOP', 'OIH', 'IEO', 'AMLP', 'USO',
            'UNG', 'BOIL', 'KOLD', 'DRIP', 'GUSH',

            # Sector ETFs - Consumer
            'XLY', 'XLP', 'VCR', 'VDC', 'XRT', 'RTH', 'FXD',
            'IBUY', 'ONLN', 'BFIT',

            # Sector ETFs - Industrial
            'XLI', 'VIS', 'IYT', 'ITA', 'XAR', 'JETS', 'PPA',

            # Sector ETFs - Materials
            'XLB', 'VAW', 'XME', 'GDX', 'GDXJ', 'SIL', 'SILJ',
            'COPX', 'PICK', 'LIT', 'REMX',

            # Sector ETFs - Real Estate
            'VNQ', 'IYR', 'XLRE', 'SCHH', 'RWR', 'ICF', 'MORT',

            # Sector ETFs - Utilities
            'XLU', 'VPU', 'IDU', 'FUTY',

            # Sector ETFs - Communication
            'XLC', 'VOX', 'IYZ', 'FCOM',

            # Bond ETFs
            'BND', 'AGG', 'TLT', 'IEF', 'SHY', 'GOVT', 'IEI',
            'LQD', 'HYG', 'JNK', 'BKLN', 'SRLN', 'FLOT',
            'TIP', 'STIP', 'SCHP', 'VTIP', 'MUB', 'HYD',
            'EMB', 'PCY', 'BWX', 'IGIB', 'VCIT', 'VCSH',
            'VGSH', 'VGIT', 'VGLT', 'BIV', 'BSV', 'BLV',
            'SCHO', 'SCHR', 'SCHZ', 'SPTS', 'SPTI', 'SPTL',
            'TMF', 'TBT', 'TBF', 'TTT',

            # Commodity ETFs
            'GLD', 'IAU', 'SLV', 'PPLT', 'PALL', 'DBA', 'DBC',
            'GSG', 'PDBC', 'DJP', 'RJI', 'BCI', 'USCI',
            'UGL', 'DUST', 'NUGT', 'JNUG', 'JDST',

            # International ETFs - Developed
            'EFA', 'VEA', 'IEFA', 'IXUS', 'VEU', 'VXUS', 'ACWI',
            'IDEV', 'SCHF', 'VGK', 'EZU', 'HEDJ', 'DBEF',
            'EWJ', 'EWG', 'EWU', 'EWC', 'EWA', 'EWL', 'EWN',

            # International ETFs - Emerging
            'EEM', 'VWO', 'IEMG', 'SCHE', 'FXI', 'KWEB', 'ASHR',
            'MCHI', 'EWZ', 'EWY', 'EWT', 'EWH', 'EWS', 'INDA',
            'EPI', 'PIN', 'THD', 'VNM', 'EPHE', 'IDX', 'TUR',
            'RSX', 'ERUS', 'GXG', 'ECH', 'EPU', 'ARGT',

            # Leveraged ETFs - Bull
            'TQQQ', 'SOXL', 'UPRO', 'SPXL', 'TNA', 'UDOW',
            'TECL', 'FAS', 'LABU', 'NAIL', 'CURE', 'NUGT',
            'JNUG', 'GUSH', 'ERX', 'RETL', 'WANT', 'DFEN',
            'DUSL', 'DPST', 'WEBL', 'FNGU', 'BULZ', 'NRGU',
            'HIBL', 'PILL', 'TPOR', 'UTSL', 'UBOT', 'EURL',
            'EDC', 'YINN', 'INDL', 'MEXX', 'DZK', 'CWEB',

            # Leveraged ETFs - Bear
            'SQQQ', 'SOXS', 'SPXU', 'SPXS', 'TZA', 'SDOW',
            'TECS', 'FAZ', 'LABD', 'DUST', 'JDST', 'DRIP',
            'ERY', 'HIBS', 'WEBS', 'FNGD', 'BERZ', 'NRGD',
            'UVXY', 'VXX', 'VIXY', 'SVXY', 'SVIX',
            'EDZ', 'YANG', 'DPK', 'EEV', 'EUM',

            # Single Stock ETFs
            'TSLL', 'TSLS', 'NVDL', 'NVDS', 'AAPD', 'AAPU',
            'MSFU', 'MSFD', 'AMZU', 'AMZD', 'NFLP', 'NFLD',
            'GOOU', 'GOOD', 'CONL', 'CONY', 'BITO', 'BITI',

            # Dividend ETFs
            'VIG', 'VYM', 'SCHD', 'DGRO', 'DVY', 'SDY', 'NOBL',
            'HDV', 'SPYD', 'SPHD', 'DHS', 'FDL', 'PEY', 'DLN',
            'VIG', 'VIGI', 'VYMI', 'IDV', 'DWX', 'SDIV',
            'DIV', 'JEPI', 'JEPQ', 'DIVO', 'QYLD', 'XYLD',
            'RYLD', 'NUSI', 'PUTW', 'PBP',

            # Thematic ETFs
            'BLOK', 'BKCH', 'DAPP', 'LEGR', 'BITQ',
            'ESPO', 'HERO', 'GAMR', 'NERD', 'SOCL',
            'CLNE', 'QCLN', 'PBW', 'TAN', 'ICLN', 'FAN',
            'DRIV', 'IDRV', 'CARZ', 'MOTO', 'EVAV',
            'UFO', 'ROKT', 'ARKX', 'SPCE',
            'AGED', 'OLD', 'GERM', 'BMED', 'EDOC',
            'AWAY', 'ACES', 'BATT', 'CNRG', 'ERTH',
            'MOON', 'PRNT', 'IZRL',

            # Fixed Income Alternatives
            'HYLD', 'HYLB', 'ANGL', 'FALN', 'SHYG', 'SJNK',
            'USHY', 'HYGV', 'HYLV', 'SPHY', 'GHYG',

            # Currency ETFs
            'UUP', 'UDN', 'FXE', 'FXY', 'FXB', 'FXC', 'FXA',
            'FXS', 'FXF', 'CYB', 'CEW',

            # Volatility ETFs
            'VXX', 'UVXY', 'SVXY', 'SVIX', 'VIXY', 'VIXM',
            'ZIV', 'VXZ', 'TVIX',

            # Specialty ETFs
            'URA', 'URNM', 'NLR', 'NUKZ',  # Nuclear/Uranium
            'ACES', 'CNRG', 'RNRG',  # Clean energy
            'DRIV', 'IDRV', 'MOTO',  # Autonomous/EVs
            'METV', 'META', 'VERS',  # Metaverse
            'AIQ', 'BOTZ', 'ROBO',  # AI/Robotics
            'ARKK', 'ARKW', 'ARKG', 'ARKF', 'ARKQ', 'ARKX',  # ARK
            'SRET', 'REM', 'MORT',  # REITs
            'PFFD', 'PFF', 'PGX', 'VRP',  # Preferred stock
        ]
        return list(set(etfs))  # Remove duplicates

    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Fetch stock data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if info.get('quoteType') == 'ETF':
                return None

            hist = ticker.history(period="1y")
            sma_50, sma_200, rsi_14 = None, None, None

            if not hist.empty and len(hist) > 0:
                closes = hist['Close']
                if len(closes) >= 50:
                    sma_50 = float(closes.tail(50).mean())
                if len(closes) >= 200:
                    sma_200 = float(closes.tail(200).mean())
                if len(closes) >= 15:
                    delta = closes.diff()
                    gain = (delta.where(delta > 0, 0)).tail(14).mean()
                    loss = (-delta.where(delta < 0, 0)).tail(14).mean()
                    if loss != 0:
                        rs = gain / loss
                        rsi_14 = float(100 - (100 / (1 + rs)))

            ex_div_date = None
            if info.get('exDividendDate'):
                try:
                    ex_div_date = datetime.fromtimestamp(
                        info['exDividendDate']
                    ).date()
                except Exception:
                    pass

            has_options = False
            try:
                has_options = len(ticker.options) > 0
            except Exception:
                pass

            return {
                'symbol': symbol,
                'company_name': info.get('longName') or info.get('shortName'),
                'exchange': info.get('exchange'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'current_price': convert_value(
                    info.get('currentPrice') or info.get('regularMarketPrice')
                ),
                'previous_close': convert_value(info.get('previousClose')),
                'open_price': convert_value(
                    info.get('open') or info.get('regularMarketOpen')
                ),
                'day_high': convert_value(
                    info.get('dayHigh') or info.get('regularMarketDayHigh')
                ),
                'day_low': convert_value(
                    info.get('dayLow') or info.get('regularMarketDayLow')
                ),
                'week_52_high': convert_value(info.get('fiftyTwoWeekHigh')),
                'week_52_low': convert_value(info.get('fiftyTwoWeekLow')),
                'volume': convert_value(
                    info.get('volume') or info.get('regularMarketVolume')
                ),
                'avg_volume_10d': convert_value(info.get('averageVolume10days')),
                'avg_volume_3m': convert_value(info.get('averageVolume')),
                'market_cap': convert_value(info.get('marketCap')),
                'shares_outstanding': convert_value(info.get('sharesOutstanding')),
                'float_shares': convert_value(info.get('floatShares')),
                'pe_ratio': convert_value(info.get('trailingPE')),
                'forward_pe': convert_value(info.get('forwardPE')),
                'peg_ratio': convert_value(info.get('pegRatio')),
                'price_to_book': convert_value(info.get('priceToBook')),
                'price_to_sales': convert_value(
                    info.get('priceToSalesTrailing12Months')
                ),
                'enterprise_value': convert_value(info.get('enterpriseValue')),
                'ev_to_ebitda': convert_value(info.get('enterpriseToEbitda')),
                'ev_to_revenue': convert_value(info.get('enterpriseToRevenue')),
                'profit_margin': convert_value(info.get('profitMargins')),
                'operating_margin': convert_value(info.get('operatingMargins')),
                'gross_margin': convert_value(info.get('grossMargins')),
                'roe': convert_value(info.get('returnOnEquity')),
                'roa': convert_value(info.get('returnOnAssets')),
                'revenue_growth': convert_value(info.get('revenueGrowth')),
                'earnings_growth': convert_value(info.get('earningsGrowth')),
                'dividend_yield': convert_value(info.get('dividendYield')),
                'dividend_rate': convert_value(info.get('dividendRate')),
                'payout_ratio': convert_value(info.get('payoutRatio')),
                'ex_dividend_date': ex_div_date,
                'total_cash': convert_value(info.get('totalCash')),
                'total_debt': convert_value(info.get('totalDebt')),
                'debt_to_equity': convert_value(info.get('debtToEquity')),
                'current_ratio': convert_value(info.get('currentRatio')),
                'free_cash_flow': convert_value(info.get('freeCashflow')),
                'beta': convert_value(info.get('beta')),
                'sma_50': sma_50,
                'sma_200': sma_200,
                'rsi_14': rsi_14,
                'target_high_price': convert_value(info.get('targetHighPrice')),
                'target_low_price': convert_value(info.get('targetLowPrice')),
                'target_mean_price': convert_value(info.get('targetMeanPrice')),
                'recommendation_key': info.get('recommendationKey'),
                'number_of_analysts': convert_value(
                    info.get('numberOfAnalystOpinions')
                ),
                'has_options': has_options,
            }
        except Exception as e:
            logger.warning(f"Error {symbol}: {e}")
            return None

    def get_etf_data(self, symbol: str) -> Optional[Dict]:
        """Fetch ETF data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if info.get('quoteType') != 'ETF':
                return None

            hist = ticker.history(period="1y")
            sma_50, sma_200, rsi_14 = None, None, None

            if not hist.empty and len(hist) > 0:
                closes = hist['Close']
                if len(closes) >= 50:
                    sma_50 = float(closes.tail(50).mean())
                if len(closes) >= 200:
                    sma_200 = float(closes.tail(200).mean())
                if len(closes) >= 15:
                    delta = closes.diff()
                    gain = (delta.where(delta > 0, 0)).tail(14).mean()
                    loss = (-delta.where(delta < 0, 0)).tail(14).mean()
                    if loss != 0:
                        rs = gain / loss
                        rsi_14 = float(100 - (100 / (1 + rs)))

            ex_div_date = None
            if info.get('exDividendDate'):
                try:
                    ex_div_date = datetime.fromtimestamp(
                        info['exDividendDate']
                    ).date()
                except Exception:
                    pass

            has_options = False
            try:
                has_options = len(ticker.options) > 0
            except Exception:
                pass

            return {
                'symbol': symbol,
                'fund_name': info.get('longName') or info.get('shortName'),
                'exchange': info.get('exchange'),
                'category': info.get('category'),
                'fund_family': info.get('fundFamily'),
                'current_price': convert_value(
                    info.get('regularMarketPrice') or info.get('previousClose')
                ),
                'previous_close': convert_value(info.get('previousClose')),
                'open_price': convert_value(info.get('regularMarketOpen')),
                'day_high': convert_value(info.get('regularMarketDayHigh')),
                'day_low': convert_value(info.get('regularMarketDayLow')),
                'week_52_high': convert_value(info.get('fiftyTwoWeekHigh')),
                'week_52_low': convert_value(info.get('fiftyTwoWeekLow')),
                'nav_price': convert_value(info.get('navPrice')),
                'volume': convert_value(info.get('regularMarketVolume')),
                'avg_volume_10d': convert_value(info.get('averageVolume10days')),
                'avg_volume_3m': convert_value(info.get('averageVolume')),
                'total_assets': convert_value(info.get('totalAssets')),
                'expense_ratio': convert_value(
                    info.get('annualReportExpenseRatio')
                ),
                'yield_ttm': convert_value(info.get('yield')),
                'ytd_return': convert_value(info.get('ytdReturn')),
                'three_year_return': convert_value(
                    info.get('threeYearAverageReturn')
                ),
                'five_year_return': convert_value(
                    info.get('fiveYearAverageReturn')
                ),
                'holdings_count': convert_value(info.get('holdingsCount')),
                'beta': convert_value(
                    info.get('beta3Year') or info.get('beta')
                ),
                'sma_50': sma_50,
                'sma_200': sma_200,
                'rsi_14': rsi_14,
                'dividend_yield': convert_value(info.get('yield')),
                'dividend_rate': convert_value(info.get('dividendRate')),
                'ex_dividend_date': ex_div_date,
                'has_options': has_options,
            }
        except Exception as e:
            logger.warning(f"Error ETF {symbol}: {e}")
            return None

    def insert_stock(self, data: Dict):
        """Insert stock data"""
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO stocks_universe (
                    symbol, company_name, exchange, sector, industry,
                    current_price, previous_close, open_price, day_high,
                    day_low, week_52_high, week_52_low, volume,
                    avg_volume_10d, avg_volume_3m, market_cap,
                    shares_outstanding, float_shares, pe_ratio, forward_pe,
                    peg_ratio, price_to_book, price_to_sales,
                    enterprise_value, ev_to_ebitda, ev_to_revenue,
                    profit_margin, operating_margin, gross_margin, roe, roa,
                    revenue_growth, earnings_growth,
                    dividend_yield, dividend_rate, payout_ratio,
                    ex_dividend_date, total_cash, total_debt,
                    debt_to_equity, current_ratio, free_cash_flow,
                    beta, sma_50, sma_200, rsi_14,
                    target_high_price, target_low_price, target_mean_price,
                    recommendation_key, number_of_analysts, has_options,
                    last_updated
                ) VALUES (
                    %(symbol)s, %(company_name)s, %(exchange)s, %(sector)s,
                    %(industry)s, %(current_price)s, %(previous_close)s,
                    %(open_price)s, %(day_high)s, %(day_low)s,
                    %(week_52_high)s, %(week_52_low)s, %(volume)s,
                    %(avg_volume_10d)s, %(avg_volume_3m)s, %(market_cap)s,
                    %(shares_outstanding)s, %(float_shares)s, %(pe_ratio)s,
                    %(forward_pe)s, %(peg_ratio)s, %(price_to_book)s,
                    %(price_to_sales)s, %(enterprise_value)s,
                    %(ev_to_ebitda)s, %(ev_to_revenue)s, %(profit_margin)s,
                    %(operating_margin)s, %(gross_margin)s, %(roe)s, %(roa)s,
                    %(revenue_growth)s, %(earnings_growth)s,
                    %(dividend_yield)s, %(dividend_rate)s, %(payout_ratio)s,
                    %(ex_dividend_date)s, %(total_cash)s, %(total_debt)s,
                    %(debt_to_equity)s, %(current_ratio)s, %(free_cash_flow)s,
                    %(beta)s, %(sma_50)s, %(sma_200)s, %(rsi_14)s,
                    %(target_high_price)s, %(target_low_price)s,
                    %(target_mean_price)s, %(recommendation_key)s,
                    %(number_of_analysts)s, %(has_options)s,
                    CURRENT_TIMESTAMP
                )
                ON CONFLICT (symbol) DO UPDATE SET
                    current_price = EXCLUDED.current_price,
                    volume = EXCLUDED.volume,
                    market_cap = EXCLUDED.market_cap,
                    sma_50 = EXCLUDED.sma_50,
                    sma_200 = EXCLUDED.sma_200,
                    rsi_14 = EXCLUDED.rsi_14,
                    last_updated = CURRENT_TIMESTAMP
            """, data)
            self.conn.commit()
        except Exception as e:
            logger.error(f"Insert error {data.get('symbol')}: {e}")
            self.conn.rollback()
        finally:
            cur.close()

    def insert_etf(self, data: Dict):
        """Insert ETF data"""
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO etfs_universe (
                    symbol, fund_name, exchange, category, fund_family,
                    current_price, previous_close, open_price, day_high,
                    day_low, week_52_high, week_52_low, nav_price,
                    volume, avg_volume_10d, avg_volume_3m, total_assets,
                    expense_ratio, yield_ttm, ytd_return,
                    three_year_return, five_year_return, holdings_count,
                    beta, sma_50, sma_200, rsi_14,
                    dividend_yield, dividend_rate, ex_dividend_date,
                    has_options, last_updated
                ) VALUES (
                    %(symbol)s, %(fund_name)s, %(exchange)s, %(category)s,
                    %(fund_family)s, %(current_price)s, %(previous_close)s,
                    %(open_price)s, %(day_high)s, %(day_low)s,
                    %(week_52_high)s, %(week_52_low)s, %(nav_price)s,
                    %(volume)s, %(avg_volume_10d)s, %(avg_volume_3m)s,
                    %(total_assets)s, %(expense_ratio)s, %(yield_ttm)s,
                    %(ytd_return)s, %(three_year_return)s,
                    %(five_year_return)s, %(holdings_count)s,
                    %(beta)s, %(sma_50)s, %(sma_200)s, %(rsi_14)s,
                    %(dividend_yield)s, %(dividend_rate)s,
                    %(ex_dividend_date)s, %(has_options)s, CURRENT_TIMESTAMP
                )
                ON CONFLICT (symbol) DO UPDATE SET
                    current_price = EXCLUDED.current_price,
                    volume = EXCLUDED.volume,
                    total_assets = EXCLUDED.total_assets,
                    sma_50 = EXCLUDED.sma_50,
                    sma_200 = EXCLUDED.sma_200,
                    rsi_14 = EXCLUDED.rsi_14,
                    last_updated = CURRENT_TIMESTAMP
            """, data)
            self.conn.commit()
        except Exception as e:
            logger.error(f"Insert ETF error {data.get('symbol')}: {e}")
            self.conn.rollback()
        finally:
            cur.close()

    def populate_stocks(self, symbols: List[str], source: str):
        """Populate stocks from a list"""
        new_symbols = [s for s in symbols if s not in self.processed_stocks]
        logger.info(f"Processing {len(new_symbols)} new stocks from {source}")

        count = 0
        for i, symbol in enumerate(new_symbols):
            try:
                data = self.get_stock_data(symbol)
                if data:
                    self.insert_stock(data)
                    self.processed_stocks.add(symbol)
                    count += 1
                    if count % 25 == 0:
                        logger.info(f"[{source}] {count}/{len(new_symbols)}")

                if (i + 1) % 5 == 0:
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error {symbol}: {e}")

        logger.info(f"Added {count} stocks from {source}")
        return count

    def populate_etfs(self, symbols: List[str], source: str):
        """Populate ETFs from a list"""
        new_symbols = [s for s in symbols if s not in self.processed_etfs]
        logger.info(f"Processing {len(new_symbols)} new ETFs from {source}")

        count = 0
        for i, symbol in enumerate(new_symbols):
            try:
                data = self.get_etf_data(symbol)
                if data:
                    self.insert_etf(data)
                    self.processed_etfs.add(symbol)
                    count += 1
                    if count % 25 == 0:
                        logger.info(f"[{source}] {count}/{len(new_symbols)}")

                if (i + 1) % 5 == 0:
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error ETF {symbol}: {e}")

        logger.info(f"Added {count} ETFs from {source}")
        return count

    def run_full_population(self):
        """Run the full population process"""
        self.get_existing_symbols()

        # Stocks
        logger.info("=" * 60)
        logger.info("POPULATING STOCKS")
        logger.info("=" * 60)

        sp500 = self.get_sp500_symbols()
        self.populate_stocks(sp500, "S&P 500")

        nasdaq100 = self.get_nasdaq100_symbols()
        self.populate_stocks(nasdaq100, "Nasdaq 100")

        dow30 = self.get_dow30_symbols()
        self.populate_stocks(dow30, "Dow 30")

        russell1000 = self.get_russell1000_symbols()
        self.populate_stocks(russell1000, "Russell 1000 additions")

        # ETFs
        logger.info("=" * 60)
        logger.info("POPULATING ETFs")
        logger.info("=" * 60)

        all_etfs = self.get_comprehensive_etf_list()
        self.populate_etfs(all_etfs, "Full ETF Universe")

        # Final count
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM stocks_universe")
        stock_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM etfs_universe")
        etf_count = cur.fetchone()[0]
        cur.close()

        logger.info("=" * 60)
        logger.info(f"COMPLETE: {stock_count} stocks, {etf_count} ETFs")
        logger.info("=" * 60)

    def close(self):
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    populator = FullUniversePopulator()
    try:
        populator.run_full_population()
    finally:
        populator.close()
