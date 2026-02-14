# cTrader Quick Start Guide

You already have `CLIENT_ID` and `CLIENT_SECRET`. Now you need to get:
1. **ACCESS_TOKEN** - Used to authenticate API requests
2. **ACCOUNT_ID** - Your cTrader account ID

## Quick Method: Use the Helper Script

I've created a helper script that automates the OAuth flow:

```bash
python get_ctrader_token.py
```

This script will:
1. ✅ Open your browser for authorization
2. ✅ Automatically capture the authorization code
3. ✅ Exchange it for an access token
4. ✅ Fetch your account ID
5. ✅ Print the values to add to `config.py`

## What Happens:

1. **Script opens browser** → You log in to cTrader
2. **You authorize the app** → Click "Authorize"
3. **Script gets token** → Automatically exchanges code for token
4. **Script gets account ID** → Fetches your account information
5. **Copy values** → Add them to `config.py`

## Manual Method (if script doesn't work)

### Step 1: Get Authorization Code

1. Open this URL in your browser (replace `YOUR_CLIENT_ID`):
   ```
   https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id=YOUR_CLIENT_ID&redirect_uri=http://localhost:8080&scope=trading&product=web
   ```

2. Log in and authorize the application

3. You'll be redirected to `http://localhost:8080?code=AUTHORIZATION_CODE`
   - Copy the `code` parameter from the URL

### Step 2: Exchange Code for Access Token

Run this command (replace values):

```bash
curl -X GET "https://openapi.ctrader.com/apps/token?grant_type=authorization_code&code=YOUR_CODE&client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET&redirect_uri=http://localhost:8080"
```

Response will be:
```json
{
  "accessToken": "your_access_token_here",
  "refreshToken": "your_refresh_token_here",
  "expiresIn": 3600
}
```

### Step 3: Get Account ID

After getting the access token, you can:
- Check your cTrader account dashboard
- Or use the test script: `python test_ctrader_data.py` (it will show account info)

## Update config.py

After getting the values, add them to `config.py`:

```python
CTRADER_ACCESS_TOKEN = 'your_access_token_here'
CTRADER_REFRESH_TOKEN = 'your_refresh_token_here'  # Optional but recommended
CTRADER_ACCOUNT_ID = 123456  # Your account ID
CTRADER_DEMO = True  # True for demo, False for live
```

## Test It

Once configured, test the connection:

```bash
python test_ctrader_data.py
```

## Important Notes

1. **Access tokens expire** (usually 1 hour)
   - Use `refresh_token` to get new access tokens
   - The helper script will show you how

2. **Redirect URI must match**
   - Must be `http://localhost:8080` (or whatever you set in your app settings)
   - Make sure it matches in your cTrader Open API app settings

3. **Demo vs Live**
   - Demo accounts: Use demo server (CTRADER_DEMO = True)
   - Live accounts: Use live server (CTRADER_DEMO = False)

## Troubleshooting

### "Authorization failed"
- Make sure redirect_uri matches your app settings
- Check that CLIENT_ID and CLIENT_SECRET are correct

### "No account found"
- Make sure you're logged into the correct cTrader account
- Check if your account has API access enabled

### "Connection timeout"
- Check your internet connection
- Try again (sometimes cTrader servers are slow)

## Next Steps

Once you have all credentials:
1. ✅ Test data fetching: `python test_ctrader_data.py`
2. ✅ Integrate with your trading strategies
3. ✅ Build CTraderBroker for order execution

