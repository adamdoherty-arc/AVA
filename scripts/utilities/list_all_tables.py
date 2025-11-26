from src.tradingview_db_manager import TradingViewDBManager

tv = TradingViewDBManager()
conn = tv.get_connection()
cur = conn.cursor()

# Get all tables
cur.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name
""")
tables = [row[0] for row in cur.fetchall()]

print(f'\nFound {len(tables)} tables in database:\n')
print(f"{'TABLE NAME':<40} {'ROW COUNT':>15}")
print('=' * 56)

for table in tables:
    try:
        cur.execute(f'SELECT COUNT(*) FROM {table}')
        count = cur.fetchone()[0]
        print(f'{table:<40} {count:>15,}')
    except Exception as e:
        print(f'{table:<40} {"ERROR":>15}')

# Check if 'stocks' table exists and show sample data
if 'stocks' in tables:
    print('\n' + '=' * 56)
    print('STOCKS TABLE DETAILS:')
    print('=' * 56)

    cur.execute('SELECT COUNT(*) FROM stocks')
    total = cur.fetchone()[0]
    print(f'\nTotal stocks in table: {total:,}')

    # Get column names
    cur.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'stocks'
        ORDER BY ordinal_position
    """)
    columns = [row[0] for row in cur.fetchall()]
    print(f'\nColumns: {", ".join(columns)}')

    # Show sample data
    cur.execute('SELECT * FROM stocks LIMIT 10')
    rows = cur.fetchall()
    print(f'\nSample data (first 10 rows):')
    for row in rows:
        print(f'  {row}')

cur.close()
conn.close()
