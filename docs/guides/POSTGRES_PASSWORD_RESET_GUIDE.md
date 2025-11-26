# PostgreSQL Password Reset Guide

## Current Issue

Your PostgreSQL service is running, but the password doesn't match your .env file.

**Expected password (from .env):** `postgres123!`

---

## Quick Fix Options

### Option 1: Reset Password via pgAdmin (Easiest - 2 minutes)

1. **Open pgAdmin 4**
   - Look for it in Start Menu: "pgAdmin 4"
   - Or go to: `C:\Program Files\PostgreSQL\16\pgAdmin 4\bin\pgAdmin4.exe`

2. **Connect to Server**
   - Expand "Servers" in left panel
   - Double-click "PostgreSQL 16"
   - Enter your CURRENT password (try common ones you use)

3. **Reset Password**
   - Right-click "postgres" user (under Login/Group Roles)
   - Click "Properties"
   - Go to "Definition" tab
   - Enter new password: `postgres123!`
   - Click "Save"

4. **Test Connection**
   ```bash
   python check_postgres.py
   ```

---

### Option 2: Reset via Command Line (Medium - 5 minutes)

If you know your current PostgreSQL password:

1. **Open Command Prompt as Administrator**

2. **Run these commands:**
   ```cmd
   cd "C:\Program Files\PostgreSQL\16\bin"

   set PGPASSWORD=YOUR_CURRENT_PASSWORD

   psql -U postgres -c "ALTER USER postgres WITH PASSWORD 'postgres123!';"
   ```

3. **Test:**
   ```bash
   cd c:\code\Magnus
   python check_postgres.py
   ```

---

### Option 3: Manual pg_hba.conf Edit (Advanced - 10 minutes)

If you don't know the current password:

1. **Run reset script as Administrator:**
   - Right-click `reset_postgres_password.bat`
   - Select "Run as Administrator"
   - Follow the prompts

2. **Manual steps (if script fails):**

   a. **Edit pg_hba.conf:**
      - Open: `C:\Program Files\PostgreSQL\16\data\pg_hba.conf`
      - Find lines with `scram-sha-256` or `md5`
      - Change to `trust` temporarily
      - Save file

   b. **Restart PostgreSQL:**
      ```cmd
      net stop postgresql-x64-16
      net start postgresql-x64-16
      ```

   c. **Reset password:**
      ```cmd
      cd "C:\Program Files\PostgreSQL\16\bin"
      psql -U postgres -c "ALTER USER postgres WITH PASSWORD 'postgres123!';"
      ```

   d. **Restore pg_hba.conf:**
      - Change `trust` back to `scram-sha-256`
      - Restart PostgreSQL again

3. **Test:**
   ```bash
   cd c:\code\Magnus
   python check_postgres.py
   ```

---

### Option 4: Find Current Password (Easiest if it exists)

Check these locations for saved passwords:

1. **pgAdmin saved passwords:**
   - `C:\Users\New User\AppData\Roaming\pgAdmin\pgpass`

2. **Windows password file:**
   - `C:\Users\New User\AppData\Roaming\postgresql\pgpass.conf`

3. **If found, update .env:**
   - Open `.env` file
   - Update `DB_PASSWORD=` with the found password
   - Test with `python check_postgres.py`

---

## After Password is Fixed

Run these commands to set up your database:

```bash
# Test connection
python check_postgres.py

# Create magnus database and apply schema
python setup_database.py

# Verify everything works
python -c "from src.magnus_local_llm import get_magnus_llm; print('LLM ready!')"
```

---

## Common PostgreSQL Passwords to Try

If you're not sure what password was set during installation:

- `postgres`
- `postgres123`
- `postgres123!`
- `admin`
- `password`
- (blank/empty)
- Your Windows password

---

## Need Help?

Current status:
- ✅ PostgreSQL 16 service is RUNNING
- ✅ Dashboard is RUNNING at http://localhost:8501
- ✅ Ollama is RUNNING
- ⏳ Qwen 32B model downloading
- ⚠️ PostgreSQL password needs reset

The password is the only blocker. Once fixed, everything else will work!
