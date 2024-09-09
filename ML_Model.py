import pandas as pd

from ml_funcs import *
import market_indicators as mi
from data_acquisition import StockObj


def prepare_dataframe(analyzed_stock_name):
    try:
        # ohlcv_data = pd.read_csv(fr'{da.OHLCV_DATA_FOLDER}\{analyzed_stock_name}_ohlcv.csv', sep=';')
        # get sentiment data

        # Get ohlcv data with indicators
        indicators_data = pd.read_csv(fr'{mi.INDICATORS_FOLDER_PATH}\{analyzed_stock_name}_indicators.csv', sep=';')
        indicators_data['date'] = pd.to_datetime(indicators_data['date'], format='%Y-%m-%d', exact=False)
        indicators_data['date'] = indicators_data['date'].dt.date
        indicators_data.set_index('date', inplace=True)
        # indicators_data.dropna(axis='index', how='any', inplace=True)

    except Exception as exc:
        print(f'Could not open data -> {exc}')
        return 0

    print(f"Dataset {analyzed_stock_name}_ohlcv.csv with length: {len(indicators_data)} acquired.")

    return indicators_data


class StockPredictionAlgo:
    def __init__(self, stock_ticker: str):
        super(StockPredictionAlgo, self).__init__()
        self.stock_name = stock_ticker
        self.stock_object = None

    def check_ticker(self):
        tmp_stock = StockObj()

def main():
    stock_name = 'WIG20.PL'
    stock_hd = prepare_dataframe(stock_name)[::-1]

    filtered_stock_data = stock_hd[['close', 'open', 'high', 'low', 'prediction', 'signal',
                                    'RSI', 'ADX', 'CCI']]

    filtered_stock_data = filtered_stock_data.dropna(inplace=False)

    # Settings
    data_split_ratio = (0.7, 0.2, 0.1)  # (train, validation, test)
    n_lookback = 5
    n_lookfor = 3
    prediction_target = 'close'
    predictors = ['open', 'high', 'low', 'close', 'prediction', 'RSI', 'ADX', 'CCI']

    # split and standardize dataset (ohlcv)
    train_df_std, val_df_std, test_df_std, train_scalar = split_and_standardize(filtered_stock_data, prediction_target,
                                                                                predictors, data_split_ratio)

    # preprocess data for training - split the dataset for the arrays of [(predictors values), predicted value]
    X_train, y_train, train_dates, X_val, y_val, val_dates, X_test, y_test, test_dates = \
        preprocess_for_training(train_df_std, val_df_std, test_df_std, n_lookback, n_lookfor)

    # Generate the dates for which prediction will be made
    dates_to_be_predicted = generate_future_working_days(test_dates, n_lookfor)

    # --------------------------------- GENERATE THE ML MODEL ---------------------------------------------------------
    ml_model, history, predictions = train_and_test(X_train, y_train, X_test, n_lookfor, (X_val, y_val))

    # --------------------------------- PREDICT THE FUTURE! ------------------------------------------------------------
    # Get the dataframe of predictors from a testing set, and remove column with predicted values
    predictors_df = test_df_std.drop(columns=train_df_std.columns[-1])
    # Get the predictor values on which the prediction of future will be performed (looking n_lookback to the past)
    predictors_of_future = predictors_df.tail(n_lookback).to_numpy()
    # Reshape the predictors to be a single point from which a prediction is being made by a model
    predictors_of_future = predictors_of_future.reshape(1, n_lookback, len(predictors_df.columns))

    future_prediction = ml_model.predict(predictors_of_future)
    # ------------------------------------------------------------------------------------------------------------------

    prediction_result = {
        'std_predicted': predictions,
        'predicted_close': rescale_data(predictions, -1, train_scalar),
        'y_test': reverse_preprocess(y_test),
        'y_test_real': rescale_data(reverse_preprocess(y_test), -1, train_scalar),
        'date': reverse_preprocess(test_dates)
    }

    future_predictions = {
        'std_predicted': future_prediction.flatten(),
        'predicted_close': rescale_data(future_prediction.flatten(), -1, train_scalar),
        'date': dates_to_be_predicted
    }

    forecast_graph = {
        'date': np.insert(dates_to_be_predicted, 0, prediction_result['date'][-1]),
        'price': np.insert(future_predictions['predicted_close'], 0, prediction_result['predicted_close'][-1])
    }

    # Calculate daily returns
    forecast_graph['daily_return'] = calculate_daily_return(forecast_graph['price'])

    # Recalculate daily returns to be in percentage
    forecast_graph['daily_return'] = np.insert(
        (forecast_graph['price'][1:] / forecast_graph['price'][:-1] - 1) * 100,
        0,
        0
    )

    forecast_df = pd.DataFrame(forecast_graph)
    forecast_df.set_index('date')
    print(f"-------------------- Forecast result ------------------------ \n{forecast_df}")

    fig, ax = plt.subplots()

    ax.plot(prediction_result['date'], prediction_result['predicted_close'], label='[test] predicted', color='blue')
    ax.plot(prediction_result['date'], prediction_result['y_test_real'], label='[test] actual', color='green')
    ax.plot(forecast_graph['date'], forecast_graph['price'], label=f'{n_lookfor}D forecast', color='yellow')

    plt.xticks(rotation='vertical')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

    return 0


# Function to calculate daily returns, assuming the first element is 0 for NaN
def calculate_daily_return(prices):
    returns = np.diff(prices) / prices[:-1] * 100
    returns = np.insert(returns, 0, 0)
    return returns


main()
