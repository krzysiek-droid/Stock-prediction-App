from datetime import datetime, timedelta, time
from dateutil.parser import parse
import os.path

import requests
from bs4 import BeautifulSoup
import pandas as pd
import pandas_datareader as dr
import urllib.request

OHLCV_DATA_FOLDER = r'D:\Python Projects\Stock prediction (ML)\OHLC data'

yf_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def create_ticker_name_dict(filepath):
    """Reads a text file with sorted tickers and names and creates a dictionary.
    Args:
        filepath: The path to the text file.

    Returns:
        A dictionary with tickers as keys and names as values.
    """
    ticker_name_dict = {}

    filename = os.path.basename(filepath).replace('.txt', '')
    ticker_tail = None
    if filename.find('-') != -1:
        ticker_tail = filename.split('-')[-1]

    with open(filepath, 'r') as file:
        for line in file:
            split_line = line.split(maxsplit=1)
            if not split_line:
                continue
            # Split the line on whitespace (assuming one space between ticker and name)
            ticker, name = split_line[0].strip(), split_line[1].strip()
            if ticker_tail is not None:
                ticker = ticker + f".{ticker_tail}"
            # Add ticker-name pair to the dictionary
            ticker_name_dict[ticker.replace("^", "")] = name
    return ticker_name_dict


class StockObj():
    tickers_folder_path = fr"D:\Python Projects\Stock prediction (ML)\stooq_tickers"

    def __init__(self):
        self.available_tickers = {}
        self.stock_name = None
        self.ohlcv = None
        self.LTD = None
        self.stock_info = None

        self.get_available_tickers()
        self.last_trading_day()

    def last_trading_day(self):
        current_date = datetime.now()
        if current_date.time() < time(17, 30):
            yesterday = current_date - timedelta(days=1)
            if yesterday.weekday() >= 4:
                days_to_subtract = (yesterday.weekday() - 4) % 7
                yesterday = current_date - timedelta(days=days_to_subtract + 1)
            self.LTD = yesterday.date()
            return yesterday.date()
        else:
            self.LTD = current_date.date()
            return current_date.date()

    def get_stooq_ohlcv(self, stock_name, ohlcv=None):
        """
        Func to acquire the stock [open, high, low, close, volume] historical data for maximum period of time (restricted
        by the data center, in this case Alpha Vantage) for given ticker in daily matter.

        Maximum period of time is constant, as for the ML model training, the longer the data - the better.

        To acquire polish index i.e. WIG or WIG20, the ticker has to be followed by .PL. This might be applicable to other
        indexes as well.
        """
        self.stock_name = stock_name

        filepath = fr'{OHLCV_DATA_FOLDER}\{self.stock_name}_ohlcv.csv'
        if os.path.exists(filepath) or ohlcv is not None:
            print(f"Reading ohlc data for {self.stock_name} from hard copy.")
            ticker_hist_df = pd.read_csv(filepath, sep=";")
            ticker_hist_df['Date'] = pd.to_datetime(ticker_hist_df['Date'])
            last_date = ticker_hist_df.iloc[0]['Date'].date()
        else:
            ticker_hist_df = None
            last_date = None

        LTD = self.last_trading_day()
        print(f"Last trading day (closed) is: {LTD}. Last day in hard-copy of dataframe is {last_date}")
        if ticker_hist_df is None or LTD != last_date:
            print(f"Fetching ohlc data for {self.stock_name}")
            ticker_hist_df = dr.DataReader(self.stock_name, 'stooq')
            if ticker_hist_df is not None:
                ticker_hist_df.dropna(inplace=True)
                ticker_hist_df.columns = ticker_hist_df.columns.str.lower()
                ticker_hist_df.to_csv(filepath, sep=';')
                ticker_hist_df.reset_index(inplace=True)
                print(f"{ticker_hist_df.index}")
                print(f"Fetching success. Date -> \n{ticker_hist_df.head(5)}")
        self.ohlcv = ticker_hist_df
        return ticker_hist_df

    def get_available_tickers(self):
        """Reads text files with sorted tickers and names from a folder and creates a dictionary.
        Args:
            folder_path: The path to the folder containing text files.

        Returns:
            A dictionary with filenames as keys and dictionaries of tickers and names as values.
        """
        folder_dict = {}
        for filename in os.listdir(self.tickers_folder_path):
            # Check if it's a text file
            if filename.endswith(".txt"):
                full_path = os.path.join(self.tickers_folder_path, filename)
                ticker_name_dict = create_ticker_name_dict(full_path)
                folder_dict[filename.replace('.txt', '').split("-")[0]] = ticker_name_dict
        self.available_tickers = folder_dict
        print(f"Available tickers groups:")
        x = 1
        for group in folder_dict.keys():
            print(f"{x}. {group}     ", end='')
            x = x + 1
        print("\n")

    def get_tickers(self, group_name: str):
        c = list(self.available_tickers[group_name].keys())
        return c

    def get_stock_info(self, stock_ticker=None):
        rows = []
        for tickers_group, tickers in self.available_tickers.items():
            for ticker, full_name in tickers.items():
                rows.append({"ticker": ticker, "full_name": full_name, "group": tickers_group})
        df = pd.DataFrame(rows)
        if stock_ticker is None:
            self.stock_info = df
            return df
        else:
            self.stock_info = df
            return df[df['ticker'] == stock_ticker]


def format_date(date_format):
    date_obj = parse(date_format)
    return date_obj.date()


if __name__ == "__main__":
    stock = '11B'
    s = StockObj()
    s.get_tickers('wse_stocks')
