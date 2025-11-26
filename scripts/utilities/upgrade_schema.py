"""Upgrade database schema for multiple expirations with delta"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host='localhost',
    database='magnus',
    user='postgres',
    password=os.getenv('DB_PASSWORD')
)

cur = conn.cursor()

print("Upgrading database schema...")

# Add delta and prob_profit columns
cur.execute("ALTER TABLE stock_premiums ADD COLUMN IF NOT EXISTS delta DECIMAL(10, 4)")
cur.execute("ALTER TABLE stock_premiums ADD COLUMN IF NOT EXISTS prob_profit DECIMAL(10, 2)")

# Drop old unique constraint
cur.execute("ALTER TABLE stock_premiums DROP CONSTRAINT IF EXISTS stock_premiums_symbol_expiration_date_strike_type_key")

# Add new unique constraint on symbol + expiration + strike (allows multiple DTEs per symbol)
cur.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = 'stock_premiums_unique'
        ) THEN
            ALTER TABLE stock_premiums
            ADD CONSTRAINT stock_premiums_unique
            UNIQUE (symbol, expiration_date, strike_price);
        END IF;
    END $$;
""")

conn.commit()

print("[OK] Database schema upgraded successfully!")
print("  - Added delta column")
print("  - Added prob_profit column")
print("  - Updated unique constraint for multiple expirations")

cur.close()
conn.close()
