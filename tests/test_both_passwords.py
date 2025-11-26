"""
Test both password variations
"""
import psycopg2

passwords = {
    'postgres123!': 'With exclamation',
    'postgres123': 'Without exclamation'
}

for pwd, desc in passwords.items():
    try:
        print(f"Testing: {desc:25} ('{pwd}') ... ", end='', flush=True)

        conn = psycopg2.connect(
            host='localhost',
            port='5432',
            database='postgres',
            user='postgres',
            password=pwd
        )

        print("SUCCESS!")
        conn.close()

    except psycopg2.OperationalError as e:
        print(f"FAILED - {str(e)[:50]}")
