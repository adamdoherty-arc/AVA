# PostgreSQL Setup Guide for Wheel Strategy

## Current Situation
- PostgreSQL 16 is installed and running
- Password authentication is failing with `postgres123!`
- You want to reuse your magnus database settings

## Step-by-Step Solution

### Option 1: Reset PostgreSQL Password (Recommended)

1. **Open Command Prompt as Administrator**
   - Right-click on Command Prompt
   - Select "Run as administrator"

2. **Navigate to PostgreSQL bin directory**
   ```cmd
   cd "C:\Program Files\PostgreSQL\16\bin"
   ```

3. **Connect to PostgreSQL using Windows authentication**
   ```cmd
   psql -U postgres -h localhost
   ```

   If this asks for password and you don't know it, try Option 2.

4. **Once connected, reset the password**
   ```sql
   ALTER USER postgres PASSWORD 'postgres123!';
   \q
   ```

### Option 2: Reset via pg_hba.conf (if Option 1 fails)

1. **Find PostgreSQL data directory**
   ```cmd
   cd "C:\Program Files\PostgreSQL\16\data"
   ```

2. **Edit pg_hba.conf**
   - Open in Notepad as Administrator
   - Find the line: `host all all 127.0.0.1/32 md5`
   - Change `md5` to `trust` temporarily
   - Save the file

3. **Restart PostgreSQL**
   ```cmd
   net stop postgresql-x64-16
   net start postgresql-x64-16
   ```

4. **Connect without password**
   ```cmd
   psql -U postgres -h localhost
   ```

5. **Reset password**
   ```sql
   ALTER USER postgres PASSWORD 'postgres123!';
   \q
   ```

6. **Revert pg_hba.conf**
   - Change `trust` back to `md5`
   - Restart PostgreSQL again

### Option 3: Use pgAdmin (Easiest if installed)

1. Open pgAdmin
2. Right-click on your PostgreSQL 16 server
3. Select Properties
4. Enter the correct password you know
5. Once connected, right-click on Login/Group Roles → postgres
6. Select Properties → Definition
7. Enter new password: `postgres123!`
8. Save

## Create the Required Database

Once you can connect, create the wheel_strategy database:

```sql
CREATE DATABASE wheel_strategy;
```

Or if you want to reuse your magnus database:
```sql
CREATE DATABASE magnus;
```

## Update .env File

Your .env file is already correct with:
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=wheel_strategy
DB_USER=postgres
DB_PASSWORD=postgres123!
```

## Test the Connection

Run this Python script to test:
```python
python test_postgres.py
```

## Using Your Magnus Database

If you want to use your existing magnus database instead:

1. Update .env file:
```
DB_NAME=magnus
DATABASE_URL=postgresql://postgres:postgres123!@localhost:5432/magnus
```

2. All your existing data from magnus will be available

## Troubleshooting

- **Port conflict**: Make sure port 5432 isn't being used by another PostgreSQL instance
- **Firewall**: Windows Firewall might block the connection
- **Service not running**: Check Services (services.msc) for postgresql-x64-16

## Next Steps

Once connected:
1. Go to "Database Scan" in the app
2. It will automatically create tables
3. Add stocks to scan
4. Watchlists will be stored in the database