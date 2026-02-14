from data_fetcher import DataFetcher

# Test EURUSD data from Yahoo Finance
fetcher = DataFetcher()
data = fetcher.fetch_data(
    symbol="EURUSD=X",
    start_date="2026-01-01",
    end_date="2026-01-15",
    interval="15m",
    to_ist=False  # Forex is typically UTC
)

print(data.head())
print(f"Data points: {len(data)}")