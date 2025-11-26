"""
Try common passwords to find PostgreSQL password
"""
import psycopg2

# Common passwords to try
passwords = [
    'postgres123!',  # From .env
    'postgres',      # Default
    'admin',
    'password',
    'postgres123',
    '123456',
    'Admin123',
    'Password123',
    '',              # Empty
]

print("Trying common PostgreSQL passwords...\n")

for pwd in passwords:
    try:
        pwd_display = f"'{pwd}'" if pwd else "(empty)"
        print(f"Trying: {pwd_display:20} ... ", end='', flush=True)

        conn = psycopg2.connect(
            host='localhost',
            port='5432',
            database='postgres',
            user='postgres',
            password=pwd
        )

        print("SUCCESS!")
        print(f"\nFound working password: {pwd}")
        print(f"\nUpdate your .env file with:")
        print(f"DB_PASSWORD={pwd}")

        conn.close()
        exit(0)

    except psycopg2.OperationalError:
        print("Failed")
        continue

print("\nNone of the common passwords worked.")
print("\nYou'll need to reset the password manually.")
print("See: POSTGRES_PASSWORD_RESET_GUIDE.md")
