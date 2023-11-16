from advanced_ta import LorentzianClassification
from talib import BBANDS, MACD
import pandas as pd
import data_acquisition as da

INDICATORS_FOLDER_PATH = fr"D:\CondaPy - Projects\Various\Stock prediction (ML)\Indicators"


def prepare_dataframe(stock_name):
    df = da.get_historical_ohlcv(stock_name)
    # Set the date columns to contain datetime objects and make it a dataframe's index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%M-%d', exact=False)
        df['Date'] = df['Date'].dt.date
        df.set_index('Date', inplace=True)
    # Change columns name to lower (required by the lorentzian classification algorithm
    df.columns = df.columns.map(lambda x: x.lower())

    return df


def calculate_lorentz(stock_data: pd.DataFrame, reverse_data=True) -> pd.DataFrame:
    # Reverse the dataframe to from latest to recent (1 row is latest, last row is recent), required for proper plotting
    if reverse_data:
        stock_data = stock_data[::-1]

    print(f"Calculating lorentz indicator for dataset length {len(stock_data)}")
    lc = LorentzianClassification(stock_data)
    lc.plot(fr'{INDICATORS_FOLDER_PATH}\{analyzed_stock}_lorentz.jpg', plot_type='line')
    if reverse_data:
        # return data in initial order
        lc.reverse_data()
    return lc.data


def calculate_indicators(stock_name, df: pd.DataFrame):
    df = calculate_lorentz(df)
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = BBANDS(df['close'][::-1], timeperiod=20)
    df['macd'], df['macdSignal'], df['macd_hist'] = MACD(df['close'][::-1])
    df.to_csv(fr'{INDICATORS_FOLDER_PATH}\{stock_name}_indicators.csv', sep=';')
    return df


if __name__ == "__main__":
    print(f"Running market indicators...")
    analyzed_stock = 'WIG20.PL'

    market_data = prepare_dataframe(analyzed_stock)

    ind = calculate_indicators(analyzed_stock, market_data)
    print(ind)

    print(f"Market indicators calculation finished...")
