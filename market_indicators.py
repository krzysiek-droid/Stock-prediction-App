from advanced_ta import LorentzianClassification
from talib import BBANDS, MACD, RSI, CCI, ADX
import pandas as pd
import data_acquisition as da

INDICATORS_FOLDER_PATH = fr"\Indicators"


def prepare_dataframe(stock_name):
    stock = da.StockObj(stock_name)
    df = stock.ohlcv
    # Set the date columns to contain datetime objects and make it a dataframe's index
    # if 'date' in df.columns:
    #     df['Date'] = pd.to_datetime(df['Date'], format='%Y-%M-%d', exact=False)
    #     df['Date'] = df['Date'].dt.date
    #     df.set_index('Date', inplace=True)
    # Change columns name to lower (required by the lorentzian classification algorithm
    df.columns = df.columns.map(lambda x: x.lower())

    return df


def calculate_lorentz(stock_data: pd.DataFrame, reverse_data=True) -> pd.DataFrame:
    # Reverse the dataframe to from latest to recent (1 row is latest, last row is recent), required for proper plotting
    if reverse_data:
        stock_data = stock_data[::-1]
    print(f"Calculating lorentz indicator for dataset length {len(stock_data)}")
    lc = LorentzianClassification(stock_data)
    # lc.plot(fr'{INDICATORS_FOLDER_PATH}\{analyzed_stock}_lorentz.jpg', plot_type='line')
    if reverse_data:
        # return data in initial order
        lc.reverse_data()

    final_df = lc.data
    boolean_columns = final_df.select_dtypes(include=bool).columns
    final_df[boolean_columns] = final_df[boolean_columns].astype(int).fillna(0)

    return final_df


def calculate_indicators(stock_name, df: pd.DataFrame):
    df = calculate_lorentz(df)
    timeperiod = 20
    close = df['close'].values[::-1]  # Convert close prices to numpy array
    high = df['high'].values[::-1]  # Convert high prices to numpy array
    low = df['low'].values[::-1]  # Convert low prices to numpy array
    print(f'{high}')

    # df['bb_upper'], df['bb_middle'], df['bb_lower'] = BBANDS(close, timeperiod=timeperiod)
    # df['macd'], df['macdSignal'], df['macd_hist'] = MACD(close)
    df['RSI'] = RSI(close)[::-1]
    df['CCI'] = CCI(high, low, close, timeperiod)[::-1]
    df['ADX'] = ADX(high, low, close, timeperiod)[::-1]
    df.to_csv(fr'{INDICATORS_FOLDER_PATH}\{stock_name}_indicators.csv', sep=';')
    return df


if __name__ == "__main__":
    print(f"Running market indicators...")
    analyzed_stock = 'WIG20.PL'

    market_data = prepare_dataframe(analyzed_stock)

    ind = calculate_indicators(analyzed_stock, market_data)
    print(ind)

    print(f"Market indicators calculation finished...")
