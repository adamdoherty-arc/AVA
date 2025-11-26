# How to Get Your TradingView Session ID

## Quick Steps:

1. **Open TradingView in Chrome**
   - Go to https://www.tradingview.com
   - Login with your account (h.adam.doherty@gmail.com)

2. **Open Chrome Developer Tools**
   - Press `F12` or right-click → "Inspect"

3. **Navigate to Cookies**
   - Click on "Application" tab (might be under >> if window is small)
   - In left sidebar: Storage → Cookies → https://www.tradingview.com

4. **Find the Session ID**
   - Look for cookie named: `sessionid`
   - The value will be a 32-character string (letters and numbers)
   - Example: `abcd1234efgh5678ijkl9012mnop3456`

5. **Copy the Session ID**
   - Double-click the value to select all
   - Copy it (Ctrl+C)

6. **Add to .env file**
   - Add this line to your .env file:
   ```
   TRADINGVIEW_SESSION_ID=your_session_id_here
   ```

## Alternative Method - Network Tab:

1. While logged into TradingView, go to your watchlists page
2. Open DevTools → Network tab
3. Refresh the page
4. Look for any request to `/api/v1/symbols_list/`
5. Click on it → Headers → Request Headers
6. Find `Cookie:` header and extract sessionid value

## Finding Your Watchlist IDs:

1. **Custom Watchlists:**
   - Go to your watchlists page
   - Click on a watchlist (like your NVDA list)
   - Look at the URL: `https://www.tradingview.com/watchlists/12345678/`
   - The 8-digit number is your watchlist ID

2. **Colored Lists:**
   - These are accessed by color name: `red`, `blue`, `green`, `orange`, `purple`, `cyan`

## Test Your Connection:

Once you have your session ID in the .env file, run:

```bash
python src/tradingview_api_sync.py
```

This will:
- Connect to TradingView using your session
- Fetch all your watchlists
- Store them in the magnus PostgreSQL database
- Show you which watchlists contain NVDA

## Important Notes:

- Session IDs expire after some time (usually days/weeks)
- If sync stops working, get a new session ID
- The session ID is like a temporary password - keep it secure
- Don't commit the .env file with your session ID to git