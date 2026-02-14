# cTrader Open API Setup Guide

This guide explains how to set up cTrader Open API to fetch forex data (EURUSD, etc.) for your trading infrastructure.

## Prerequisites

1. **cTrader Account**: You need a cTrader account (demo or live)
   - FTMO provides cTrader accounts
   - You can also create a demo account at any cTrader broker

2. **Open API Application**: Create an application in the cTrader Open API portal

## Step 1: Install cTrader Open API SDK

```bash
pip install ctrader-open-api
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Step 2: Create Open API Application

1. Go to [cTrader Open API Portal](https://openapi.ctrader.com/)
2. Log in with your cTrader account
3. Create a new application
4. You'll receive:
   - **Client ID**
   - **Client Secret**
   - Set up a **Redirect URI** (e.g., `http://localhost:8080`)

## Step 3: Get Access Token

The cTrader Open API uses OAuth 2.0 authentication. You need to:

1. **Get Authorization Code**:
   - Redirect user to authorization URL:
   ```
   https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope=trading&product=web
   ```
   - User authorizes and gets redirected with a `code` parameter

2. **Exchange Code for Access Token**:
   ```bash
   curl -X GET "https://openapi.ctrader.com/apps/token?grant_type=authorization_code&code={CODE}&client_id={CLIENT_ID}&client_secret={CLIENT_SECRET}&redirect_uri={REDIRECT_URI}"
   ```
   
   Response includes:
   - `accessToken`
   - `refreshToken`
   - `expiresIn`

3. **Get Account ID**:
   - After authentication, you'll get a list of accounts
   - Note the `ctidTraderAccountId` for the account you want to use

## Step 4: Set Environment Variables

Set these environment variables with your credentials:

```bash
export CTRADER_CLIENT_ID='your_client_id'
export CTRADER_CLIENT_SECRET='your_client_secret'
export CTRADER_ACCESS_TOKEN='your_access_token'
export CTRADER_ACCOUNT_ID='your_account_id'
export CTRADER_DEMO='true'  # or 'false' for live
```

Or create a `.env` file (make sure it's in `.gitignore`):

```bash
CTRADER_CLIENT_ID=your_client_id
CTRADER_CLIENT_SECRET=your_client_secret
CTRADER_ACCESS_TOKEN=your_access_token
CTRADER_ACCOUNT_ID=your_account_id
CTRADER_DEMO=true
```

## Step 5: Test the Connection

Run the test script:

```bash
python test_ctrader_data.py
```

This will:
- Fetch EURUSD data for the last 7 days
- Display data summary
- Test current price fetching

## Usage in Your Code

```python
from data_fetcher import CTraderDataFetcher
from datetime import datetime, timedelta

# Initialize fetcher
fetcher = CTraderDataFetcher(
    client_id='your_client_id',
    client_secret='your_client_secret',
    access_token='your_access_token',
    account_id=123456,  # Your account ID
    demo=True  # or False for live
)

# Fetch historical data
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

data = fetcher.fetch_historical_data(
    symbol='EURUSD',
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d'),
    interval='15m'  # Options: '1m', '5m', '15m', '30m', '1h', '4h', '1d'
)

print(data.head())
```

## Supported Intervals

- `1m` - 1 minute
- `5m` - 5 minutes
- `15m` - 15 minutes
- `30m` - 30 minutes
- `1h` - 1 hour
- `4h` - 4 hours
- `1d` - 1 day

## Supported Symbols

Common forex pairs:
- `EURUSD` - Euro/US Dollar
- `GBPUSD` - British Pound/US Dollar
- `USDJPY` - US Dollar/Japanese Yen
- `AUDUSD` - Australian Dollar/US Dollar
- `USDCAD` - US Dollar/Canadian Dollar
- And many more...

## Limitations

1. **Rate Limits**:
   - Historical data: ~5 requests/second
   - Non-historical data: ~50 requests/second

2. **Historical Data**:
   - Tick data requests limited to one week at a time
   - Bar data can go further back

3. **Access Token Expiry**:
   - Access tokens expire (check `expiresIn`)
   - Use refresh token to get new access token

## Troubleshooting

### "Symbol not found"
- Make sure the symbol name matches exactly (e.g., `EURUSD` not `EURUSD=X`)
- Check if the symbol is available on your broker

### "Authentication failed"
- Verify your credentials are correct
- Check if access token has expired
- Ensure account has API access enabled

### "Connection timeout"
- Check your internet connection
- Verify you're using the correct endpoint (demo vs live)
- Check firewall settings

## Resources

- [cTrader Open API Documentation](https://help.ctrader.com/open-api/)
- [Python SDK GitHub](https://github.com/spotware/OpenApiPy)
- [cTrader Open API Portal](https://openapi.ctrader.com/)

## Next Steps

Once you have data fetching working:
1. Create `CTraderBroker` class for order execution
2. Create `CTraderTradingEngine` to integrate with your strategies
3. Test with demo account before going live

