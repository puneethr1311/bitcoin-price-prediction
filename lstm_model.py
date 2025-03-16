import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from itertools import product

def prepare_lstm_data(data, look_back):
    
    print("Prepare data for LSTM with a given look-back period.")
    
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2):
    
    print("Build an LSTM model with configurable hyperparameters.")
    
    model = Sequential([
        LSTM(lstm_units, activation='relu', return_sequences=True, input_shape=input_shape),
        # Bidirectional(LSTM(lstm_units, activation='relu', return_sequences=True), input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units, activation='relu'),
        # Bidirectional(LSTM(lstm_units, activation='relu')),
        Dropout(dropout_rate),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_evaluate_lstm(
    data,
    look_back=30,
    epochs=200,
    batch_size=32,
    lstm_units=50,
    dropout_rate=0.2,
    scaler_type="MinMaxScaler",
):

    print("Train and evaluate the LSTM model with hyperparameter tuning and flexible data scaling.")

    # Step 1: Normalize the data
    scaler = {
        "MinMaxScaler": MinMaxScaler(),
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler()
    }[scaler_type]
    scaled_data = scaler.fit_transform(data)
    # scaled_data = scaler.fit_transform(data.reshape(-1, 1))  # Ensure data is 2D

    # Step 2: Prepare LSTM data
    X, y = prepare_lstm_data(scaled_data, look_back)
    # X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape to (samples, timesteps, features)

    train_size = int(len(X) * 0.8)
    val_size = int(len(X) * 0.1)
    # test_size = len(X) - (train_size + val_size)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    # X_test, y_test = X[-test_size:], y[-test_size:]  # Ensure test set is properly indexed

    # Step 3: Build and train the model
    model = build_lstm_model((look_back, 1), lstm_units, dropout_rate)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        # callbacks=[reduce_lr]  # Learning rate scheduler
    )

    # Step 4: Evaluate on test data
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    # y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)

    metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
    print("LSTM Model Metrics:", metrics)

    return model, metrics, scaler, y_pred_rescaled.flatten(), y_test_rescaled.flatten()

def perform_hyperparameter_tuning(data, look_back_values, epochs_values, batch_size_values, lstm_units_values, dropout_rate_values):
    
    print("Perform hyperparameter tuning and evaluate different configurations.")
    
    best_metrics = None
    best_config = None

    for look_back, epochs, batch_size, lstm_units, dropout_rate in product(
        look_back_values, epochs_values, batch_size_values, lstm_units_values, dropout_rate_values
    ):
        print(f"Testing Configuration: look_back={look_back}, epochs={epochs}, batch_size={batch_size}, lstm_units={lstm_units}, dropout_rate={dropout_rate}")
        _, metrics, _, _, _ = train_and_evaluate_lstm(
            data,
            look_back=look_back,
            epochs=epochs,
            batch_size=batch_size,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate
        )
        if best_metrics is None or metrics["RMSE"] < best_metrics["RMSE"]:
            best_metrics = metrics
            best_config = {
                "look_back": look_back,
                "epochs": epochs,
                "batch_size": batch_size,
                "lstm_units": lstm_units,
                "dropout_rate": dropout_rate
            }

    print("Best Configuration:", best_config)
    print("Best Metrics:", best_metrics)
    return best_config, best_metrics