from datetime import datetime, timedelta, time
from dateutil.parser import parse
import os.path

import pandas as pd
import pandas_datareader as dr

OHLCV_DATA_FOLDER = r'D:\CondaPy - Projects\Various\Stock prediction (ML)\OHLC data'

yf_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def last_trading_day():
    current_date = datetime.now()
    if current_date.time() < time(16, 0):
        yesterday = current_date - timedelta(days=1)
        if yesterday.weekday() >= 4:
            days_to_subtract = (yesterday.weekday() - 4) % 7
            yesterday = current_date - timedelta(days=days_to_subtract + 1)
        return yesterday.date()
    else:
        return current_date.date()


def get_historical_ohlcv(ticker: str):
    """
    Func to acquire the stock [open, high, low, close, volume] historical data for maximum period of time (restricted
    by the data center, in this case stooq) for given ticker in daily matter.

    Maximum period of time is constant, as for the ML model training, the longer the data - the better.

    To acquire polish index i.e. WIG or WIG20, the ticker has to be followed by .PL. This might be applicable to other
    indexes as well.
    """
    filepath = fr'{OHLCV_DATA_FOLDER}\{ticker}_ohlcv.csv'
    if os.path.exists(filepath):
        print(f"Reading ohlc data for {ticker} from hard copy.")
        ticker_hist_df = pd.read_csv(filepath, sep=";")
        last_date = format_date(ticker_hist_df.iloc[0]['Date'])
    else:
        ticker_hist_df = None
        last_date = None

    LTD = last_trading_day()
    print(f"Last trading day (closed) is: {LTD}. Last day in hard-copy of dataframe is {last_date}")
    if ticker_hist_df is None or LTD != last_date:
        print(f"Fetching ohlc data for {ticker}")
        ticker_hist_df = dr.get_data_stooq(ticker)
        ticker_hist_df.dropna()
        ticker_hist_df.to_csv(filepath, sep=';')
        print(f"Fetching success. Columns -> {ticker_hist_df.columns}")

    return ticker_hist_df


def format_date(yf_date):
    date_obj = parse(yf_date)
    return date_obj.date()


if __name__ == "__main__":
    stock = 'WIG20.PL'

    s = get_historical_ohlcv(stock)
    print(s)
