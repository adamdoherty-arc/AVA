# Fix pgAdmin Password After Change

## Quick Solution

### Method 1: Update Saved Password in pgAdmin

1. **Open pgAdmin**

2. **Find your PostgreSQL 16 server** in the left panel
   - It will show a red X or disconnected icon

3. **Right-click on the server** → **Properties**

4. **Go to Connection tab**
   - Keep everything the same
   - **Password field**: Enter your new password: `postgres123!`
   - **Save password**: Check this box

5. **Click Save**

6. **Try to connect** by clicking on the server
   - It should now connect successfully

### Method 2: Remove and Re-add Server

If Method 1 doesn't work:

1. **Right-click on the server** → **Remove Server**

2. **Right-click on "Servers"** → **Register** → **Server**

3. **General tab:**
   - Name: `PostgreSQL 16`

4. **Connection tab:**
   - Host: `localhost`
   - Port: `5432`
   - Maintenance database: `postgres`
   - Username: `postgres`
   - Password: `postgres123!`
   - Save password: ✓ Check this

5. **Click Save**

### Method 3: Clear pgAdmin Saved Passwords

If still having issues:

1. **Close pgAdmin**

2. **Delete pgAdmin config** (Windows):
   ```
   C:\Users\%USERNAME%\AppData\Roaming\pgAdmin\
   ```
   Delete the `pgpass.conf` file

3. **Restart pgAdmin**
   - It will ask for password again
   - Enter: `postgres123!`

## Verify Connection

Once connected in pgAdmin:

1. Expand your server
2. Expand Databases
3. You should see:
   - `magnus` - Your existing database (we're using this)
   - `postgres` - Default database
   - `stacks` - Your other database
   - `aistock` - Your AI stock database
   - `wheel_strategy` - New database we created

## Current Configuration

Your app is now configured to use:
- **Database**: `magnus`
- **User**: `postgres`
- **Password**: `postgres123!`
- **Port**: `5432`
- **Host**: `localhost`

## Test the Connection

Run this to verify everything works:
```bash
python test_pg_simple.py
```

## Troubleshooting

### If pgAdmin keeps failing:

1. **Use psql command line instead:**
   ```bash
   psql -U postgres -h localhost -d magnus
   ```
   Password: `postgres123!`

2. **Alternative GUI Tools:**
   - DBeaver (free)
   - TablePlus
   - HeidiSQL

3. **Check pgAdmin logs:**
   - Help → View Log File
   - Look for authentication errors

### Common Issues:

- **"FATAL: password authentication failed"**
  - The saved password in pgAdmin is old
  - Use Method 1 or 2 above

- **"Connection refused"**
  - PostgreSQL service might be stopped
  - Run: `net start postgresql-x64-16`

- **"Timeout expired"**
  - Firewall blocking connection
  - Check Windows Firewall settings