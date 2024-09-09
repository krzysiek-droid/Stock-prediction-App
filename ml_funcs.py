import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import timedelta, date
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
import seaborn as sns


def run_LSTM(training_features, training_target, forecast_steps, validation_data_tuple,
             nodes=20, MAX_EPOCHS=50):
    early_stopping = keras.callbacks.EarlyStopping(monitor='mean_absolute_error', patience=10, mode='min')

    ml_model = Sequential()

    # Layer structure
    ml_model.add(LSTM(units=nodes, return_sequences=False))
    ml_model.add(Dropout(0.2))
    ml_model.add(Dense(forecast_steps, kernel_initializer=keras.initializers.zeros()))

    # compile the neural net model (LSTM)
    ml_model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(),
                     metrics=[keras.metrics.MeanAbsoluteError()])

    history = ml_model.fit(training_features, training_target, epochs=MAX_EPOCHS, validation_data=validation_data_tuple,
                           callbacks=[early_stopping])
    ml_model.summary()

    return ml_model, history


def train_and_test(training_predictors_set, training_predictions_set, testing_predictors_set,
                   n_lookfor, validation_set: tuple, nodes_multiplier=25, MAX_EPOCHS=30):
    nodes = len(training_predictors_set[-1]) * nodes_multiplier

    ml_model, history = run_LSTM(training_predictors_set, training_predictions_set, n_lookfor, validation_set, nodes,
                                 MAX_EPOCHS)

    # predict the past (use the model to predict the points in time that we can validate)
    predicted_values = ml_model.predict(testing_predictors_set)
    predicted_values = reverse_preprocess(predicted_values)

    # reset the model states, to prepare it for prediction of future
    ml_model.reset_states()

    return ml_model, history, predicted_values


def split_data(features_df: pd.DataFrame, y_target: pd.DataFrame, split_ratios: tuple):
    """
        split ratio: a tuple containing the fractions of dataframe length designated for training, validation and testing
    """

    train_val_index = int(len(features_df) * split_ratios[0])
    val_test_index = train_val_index + int(len(features_df) * split_ratios[1])

    X_train = features_df.to_numpy()[:train_val_index]
    y_train = y_target.to_numpy()[:train_val_index]
    train_dates = features_df.index.values[:train_val_index]
    train_df = pd.DataFrame(X_train, columns=features_df.columns, index=train_dates)
    train_df['y_target'] = y_train

    X_val = features_df.to_numpy()[train_val_index:val_test_index]
    y_val = y_target.to_numpy()[train_val_index:val_test_index]
    val_dates = features_df.index.values[train_val_index:val_test_index]
    val_df = pd.DataFrame(X_val, columns=features_df.columns, index=val_dates)
    val_df['y_target'] = y_val

    X_test = features_df.to_numpy()[val_test_index:]
    y_test = y_target.to_numpy()[val_test_index:]
    test_dates = y_target.index.values[val_test_index:]
    test_df = pd.DataFrame(X_test, columns=features_df.columns, index=test_dates)
    test_df['y_target'] = y_test

    print(f"Train length: {len(X_train)}, Validation length: {len(X_val)}, Test length: {len(X_test)}")
    print(f"Train index: {train_val_index}, Validation index: {val_test_index}")
    return train_df, val_df, test_df


def split_and_standardize(ohlcv_data: pd.DataFrame, prediction_target_column_name: str,
                          predictors_column_names: list, data_split_ratio: tuple):
    """Prepare dataset, meaning splitting with the 'data_split_ratio' and standardizing"""
    stock_prices = ohlcv_data[[prediction_target_column_name]]
    features_df = ohlcv_data[predictors_column_names]

    # Split the data for the training, validation and test groups
    train_df, val_df, test_df = split_data(features_df, stock_prices, data_split_ratio)

    # Standardize the data
    train_df_std, train_scalar = standardize_features(train_df)
    val_df_std, _ = standardize_features(val_df, train_scalar)
    test_df_std, _ = standardize_features(test_df, train_scalar)

    return train_df_std, val_df_std, test_df_std, train_scalar


def preprocess_for_training(training_set, validation_set, testing_set, n_lookback, n_lookfor):
    """ preprocessing the ohlcv sets (training, validation and testing) for training meaning splitting each of them into
    the sets of ((predictors values*lookback timestamps), resulting target*lookfor timestamps). OHLCV sets must be
    standardized !
    """
    dataset_columns = training_set.columns

    # remove the last columns form the datasets, which is reserved for predicted value
    X_train, y_train, train_dates = preprocess_data(training_set.drop(columns=dataset_columns[-1]).to_numpy(),
                                                    training_set[dataset_columns[-1]].to_numpy(),
                                                    training_set.index.values, n_lookback, n_lookfor)
    X_val, y_val, val_dates = preprocess_data(validation_set.drop(columns=dataset_columns[-1]).to_numpy(),
                                              validation_set[dataset_columns[-1]].to_numpy(),
                                              validation_set.index.values, n_lookback, n_lookfor)
    X_test, y_test, test_dates = preprocess_data(testing_set.drop(columns=dataset_columns[-1]).to_numpy(),
                                                 testing_set[dataset_columns[-1]].to_numpy(),
                                                 testing_set.index.values, n_lookback, n_lookfor)

    return X_train, y_train, train_dates, X_val, y_val, val_dates, X_test, y_test, test_dates


def plot_features_standardized(features_df):
    df_std = features_df.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(features_df.keys(), rotation=90)
    plt.show()


def preprocess_data(X_data, y_data, dates_data, data_seq: int, days_to_predict=5):
    X, y, y_dates = [], [], []
    for i in range(data_seq, len(X_data) - days_to_predict + 1):
        X.append(X_data[i - data_seq: i])
        y.append(y_data[i: i + days_to_predict])
        y_dates.append(dates_data[i: i + days_to_predict])

    return np.array(X), np.array(y), np.array(y_dates)


def generate_future_working_days(testing_dates: np.array, days_into_future) -> np.array:
    """Take the set of last days, that was used to test the latest prediction (for last trading day)"""
    print(f"Last day tested: -> {testing_dates[-1][-1]}")
    current_date = testing_dates[-1][-1] + timedelta(days=1)
    working_days = []
    while len(working_days) < days_into_future:
        # Skip weekends (Saturday and Sunday)
        if current_date.weekday() < 5:  # Monday to Friday (0 to 4)
            working_days.append(current_date)
        # Move to the next day
        current_date += timedelta(days=1)

    return np.array(working_days)


def reverse_preprocess(array: np.array):
    # print(f"Reversing preprocessing of data shape {array.shape}...", end='')
    result = []
    for sequence in array:
        result.append(sequence[0])
    result.extend(array[-1][1:])
    # print(f"Reversed. Length of new array: {len(result)}, last record: {result[-1]}")
    return np.array(result)


def standardize_features(features: pd.DataFrame, pretrained_scaler=None):
    if pretrained_scaler is None:
        if len(features.columns) > 1:
            scaler = StandardScaler()
            cols = features.columns
            features[cols] = scaler.fit_transform(features[cols])
        else:
            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(features.values.reshape(-1, 1))
            features = pd.DataFrame(standardized_data, index=features.index, columns=features.columns)
    else:
        if len(features.columns) > 1:
            scaler = pretrained_scaler
            cols = features.columns
            features[cols] = scaler.transform(features[cols])
        else:
            scaler = pretrained_scaler
            standardized_data = scaler.transform(features.values.reshape(-1, 1))
            features = pd.DataFrame(standardized_data, index=features.index, columns=features.columns)
    return features, scaler


def rescale_data(standardized_data_column, column_index, scaler: StandardScaler):
    initial_data_width = len(scaler.mean_)
    tmp_array = np.random.uniform(0, 1, (len(standardized_data_column), initial_data_width))
    tmp_df = pd.DataFrame(tmp_array, columns=scaler.get_feature_names_out())
    column_toSwap = tmp_df.columns[column_index]
    tmp_df[column_toSwap] = standardized_data_column
    rescaled_df = scaler.inverse_transform(tmp_df)
    tmp_df = pd.DataFrame(rescaled_df, columns=scaler.get_feature_names_out())

    return tmp_df[column_toSwap].values


def calculate_daily_return(price_sequence: np.array) -> np.array:
    d_return = (price_sequence[1:] - price_sequence[:-1]) / price_sequence[:-1] * 100
    length_diff = len(price_sequence) - len(d_return)
    d_return = np.round(d_return, 2)
    for i in range(length_diff):
        d_return = np.insert(d_return, 0, 0)
    return d_return
