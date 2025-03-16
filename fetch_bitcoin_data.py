import yfinance as yf

def fetch_bitcoin_data(start_date='2000-01-01', end_date='2024-10-01'):
    print('Fetching Bitcoin historical data...')
    btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
    btc_data.reset_index(inplace=True)
    btc_data.to_csv('data/bitcoin_data.csv', index=False)
    print("Data fetched successfully and saved to 'data/bitcoin_data.csv'")
    return btc_data