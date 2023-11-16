import matplotlib.pyplot as plt
import pandas as pd

from ml_funcs import *
import market_indicators as mi


def prepare_dataframe(analyzed_stock_name):
    try:
        # ohlcv_data = pd.read_csv(fr'{da.OHLCV_DATA_FOLDER}\{analyzed_stock_name}_ohlcv.csv', sep=';')
        # get sentiment data

        # Get ohlcv data with indicators
        indicators_data = pd.read_csv(fr'{mi.INDICATORS_FOLDER_PATH}\{analyzed_stock_name}_indicators.csv', sep=';')
        indicators_data['Date'] = pd.to_datetime(indicators_data['Date'], format='%Y-%m-%d', exact=False)
        indicators_data['Date'] = indicators_data['Date'].dt.date
        indicators_data.set_index('Date', inplace=True)
        # indicators_data.dropna(axis='index', how='any', inplace=True)

    except Exception as exc:
        print(f'Could not open data -> {exc}')
        return 0

    print(f"Dataset {analyzed_stock_name}_ohlcv.csv with length: {len(indicators_data)} acquired.")

    return indicators_data


def main():
    stock_name = 'WIG20.PL'
    stock_hd = prepare_dataframe(stock_name)[::-1]

    data_split_ratio = (0.7, 0.2, 0.1)  # (train, validation, test)
    n_lookback = 14
    n_lookfor = 5

    filtered_stock_data = stock_hd[['close', 'open', 'prediction', 'bb_upper', 'bb_middle',
                                    'bb_lower', 'macd', 'macdSignal', 'macd_hist']]

    filtered_stock_data.dropna(inplace=True)
    print(filtered_stock_data)
    stock_prices = filtered_stock_data[['close']]
    features_df = filtered_stock_data[['open', 'prediction', 'bb_upper', 'bb_middle',
                                       'bb_lower', 'macd', 'macdSignal', 'macd_hist']]

    # Split the data for the training, validation and test groups
    train_df, val_df, test_df = split_data(features_df, stock_prices, data_split_ratio)

    train_df_std, train_scalar = standardize_features(train_df)
    val_df_std, _ = standardize_features(val_df, train_scalar)
    test_df_std, _ = standardize_features(test_df, train_scalar)

    dataset_columns = train_df.columns

    standardization_parameters_df = pd.DataFrame([train_scalar.mean_, train_scalar.var_],
                                                 columns=train_df_std.columns,
                                                 index=["Mean", "Variance"])

    X_train, y_train, train_dates = preprocess_data(train_df_std.drop(columns=dataset_columns[-1]).to_numpy(),
                                                    train_df_std[dataset_columns[-1]].to_numpy(),
                                                    train_df_std.index.values, n_lookback, n_lookfor)
    X_val, y_val, val_dates = preprocess_data(val_df_std.drop(columns=dataset_columns[-1]).to_numpy(),
                                              val_df_std[dataset_columns[-1]].to_numpy(),
                                              val_df_std.index.values, n_lookback, n_lookfor)
    X_test, y_test, test_dates = preprocess_data(test_df_std.drop(columns=dataset_columns[-1]).to_numpy(),
                                                 test_df_std[dataset_columns[-1]].to_numpy(),
                                                 test_df_std.index.values, n_lookback, n_lookfor)

    test_df_std = test_df_std.drop(columns=dataset_columns[-1])
    X_for_future = test_df_std.tail(n_lookback).to_numpy()
    predicted_dates = generate_future_working_days(test_df_std.index.values[-1], n_lookfor)
    X_for_future = X_for_future.reshape(1, n_lookback, len(test_df_std.columns))

    ml_model, history = run_LSTM(X_train, y_train, n_lookfor, (X_val, y_val), nodes=len(dataset_columns) * 10,
                                 MAX_EPOCHS=50)

    predictions = ml_model.predict(X_test)
    predictions = reverse_preprocess(predictions)

    ml_model.reset_states()

    future_prediction = ml_model.predict(X_for_future)

    prediction_result = {
        'std_predicted': predictions,
        'predicted_close': rescale_data(predictions, -1, train_scalar),
        'y_test': reverse_preprocess(y_test),
        'y_test_real': rescale_data(reverse_preprocess(y_test), -1, train_scalar),
        'Date': reverse_preprocess(test_dates)
    }

    future_predictions = {
        'std_predicted': future_prediction.flatten(),
        'predicted_close': rescale_data(future_prediction.flatten(), -1, train_scalar),
        'Date': predicted_dates
    }


    forecast_graph = {
        'Date': np.insert(predicted_dates, 0, prediction_result['Date'][-1]),
        'price': np.insert(future_predictions['predicted_close'], 0, prediction_result['predicted_close'][-1])
    }
    forecast_graph['daily_return'] = calculate_daily_return(forecast_graph['price'])
    forecast_df = pd.DataFrame(forecast_graph)
    forecast_df.set_index('Date')
    print(f"-------------------- Forecast result ------------------------ \n{forecast_df}")


    fig, ax = plt.subplots()

    ax.plot(prediction_result['Date'], prediction_result['predicted_close'], label='[test] predicted', color='blue')
    ax.plot(prediction_result['Date'], prediction_result['y_test_real'], label='[test] actual', color='green')
    ax.plot(forecast_graph['Date'], forecast_graph['price'], label=f'{n_lookfor}D forecast', color='yellow')

    plt.xticks(rotation='vertical')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

    return 0


main()
