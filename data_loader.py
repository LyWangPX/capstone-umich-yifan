# Author: Yifan Wang
import yfinance as yf
import pandas as pd


class YahooDownloader:
    def fetch_data(self, symbols_list, start, end):
        data_dict = {}
        
        for symbol in symbols_list:
            try:
                df = yf.download(symbol, start=start, end=end, progress=False)
                if not df.empty:
                    df = df.dropna()
                    data_dict[symbol] = df
                    print(f"Downloaded {symbol}: {len(df)} rows")
                else:
                    print(f"No data for {symbol}")
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
        
        return data_dict
