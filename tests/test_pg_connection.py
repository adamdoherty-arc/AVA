import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

passwords_to_try = [
    os.getenv('DB_PASSWORD', 'postgres123!'),
    'postgres123!',
    'postgres',
    ''
]

for pwd in passwords_to_try:
    try:
        print(f"Trying password: {'(empty)' if pwd == '' else '****'}")
        conn = psycopg2.connect(
            host='localhost',
            port='5432',
            database='postgres',
            user='postgres',
            password=pwd
        )
        print(f"✅ SUCCESS! Password is: {pwd}")
        conn.close()
        break
    except Exception as e:
        print(f"   Failed: {str(e)[:50]}")
else:
    print("\n❌ None of the passwords worked")
    print("You may need to reset the PostgreSQL password")
