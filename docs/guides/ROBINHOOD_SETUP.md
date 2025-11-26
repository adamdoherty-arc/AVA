# Robinhood Integration Setup Guide

## Quick Start

The system is now configured to automatically connect to Robinhood using your credentials from the `.env` file!

### First Time Setup (One-time only)

1. **Run the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

2. **Navigate to the Positions page**
   - The system will attempt to auto-connect using your saved credentials
   - If this is your first time, you'll need to complete MFA verification

3. **Complete MFA Verification:**
   - When prompted, check your phone/email for the verification request
   - Approve the login or enter the code if prompted
   - The session will be saved automatically

### Subsequent Logins (No MFA needed!)

After your first successful login, the system will:
- Automatically use your cached session
- Connect without requiring MFA
- Load your positions immediately

## Features

### Auto-Connect
- Credentials are loaded from `.env` file automatically
- No need to type username/password each time
- Session caching prevents repeated MFA requests

### Session Management
- Sessions are cached in your home directory
- Valid for 24 hours by default
- Automatically refreshed when needed

## Troubleshooting

### If Auto-Connect Fails

1. **Check your .env file has:**
   ```
   ROBINHOOD_USERNAME=your_username
   ROBINHOOD_PASSWORD=your_password
   ```

2. **For first-time login:**
   - You MUST complete MFA verification
   - Check your email/phone for the verification request
   - This only happens once

3. **Clear cached session if needed:**
   ```bash
   python -c "from pathlib import Path; import os; p=Path.home()/'.robinhood_session.pickle'; os.remove(p) if p.exists() else None; print('Cache cleared')"
   ```

### Manual Testing

Test the connection directly:
```bash
python src/robinhood_auth.py
```

This will:
- Attempt to connect with your saved credentials
- Show detailed error messages if something fails
- Save the session for future use

## Security Notes

- Credentials are never stored in the session cache
- Only authentication tokens are cached
- Sessions expire after 24 hours for security
- You can disconnect anytime from the dashboard

## Dashboard Features

Once connected, you can:
- View all your stock positions
- See open options (puts/calls)
- Track wheel strategy positions
- Monitor account balance and buying power
- Identify covered call opportunities
- Track cash-secured puts

## How It Works

1. **Initial Connection:**
   - Uses credentials from `.env`
   - Prompts for MFA if needed (first time only)
   - Saves session token

2. **Cached Sessions:**
   - Subsequent connections use saved token
   - No MFA required
   - Automatic and instant connection

3. **Data Refresh:**
   - Click "Refresh Positions" to update data
   - Positions update in real-time
   - Account values are current

## Next Steps

1. Start the dashboard: `streamlit run dashboard.py`
2. Go to the Positions page
3. Watch as it auto-connects with your saved credentials!
4. Your real positions will load automatically

No more typing credentials every time!