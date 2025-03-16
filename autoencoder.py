from keras.models import Model
from keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def apply_autoencoder(data):
    print('Applying Autoencoder for Dimensionality Reduction...')

    # Ensure the 'Adj Close' column is dropped explicitly
    if 'Adj Close' in data.columns:
        data = data.drop(columns=['Adj Close'])

    # Retain the 'Date' column separately
    date_column = data['Date']

    # Ensure the 'Date' column contains only the date (no time component)
    date_column = pd.to_datetime(date_column).dt.date

    # Select numeric columns for dimensionality reduction
    numeric_columns = [ 'Close', 'High', 'Low', 'Open', 'Volume']
    data_to_encode = data[numeric_columns]

    # Normalize the numeric columns
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_to_encode)

    # Define the Autoencoder model
    input_dim = scaled_data.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(64, activation='relu')(input_layer)
    encoder = Dense(32, activation='relu')(encoder)
    encoder = Dense(16, activation='relu')(encoder)   # Latent features: 16
    decoder = Dense(32, activation='relu')(encoder)
    decoder = Dense(64, activation='relu')(decoder)
    decoder = Dense(input_dim, activation="sigmoid")(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the Autoencoder
    autoencoder.fit(scaled_data, scaled_data, epochs=200, batch_size=32, shuffle=True)

    # Extract the reduced dataset using the encoder
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    reduced_data = encoder_model.predict(scaled_data)

    # Map latent features back to the original numeric columns
    num_latent_features = reduced_data.shape[1]
    num_target_columns = len(numeric_columns)

    # Divide latent features evenly into groups for each target column
    features_per_column = num_latent_features // num_target_columns
    grouped_features = [
        reduced_data[:, i * features_per_column : (i + 1) * features_per_column]
        for i in range(num_target_columns)
    ]

    # Average features within each group
    reduced_mapped_data = np.hstack([
        np.mean(group, axis=1, keepdims=True) for group in grouped_features
    ])

    # Convert reduced data back to a DataFrame
    reduced_df = pd.DataFrame(reduced_mapped_data, columns=numeric_columns, index=data.index)

    # Reattach the 'Date' column
    reduced_df.insert(0, 'Date', date_column)
    
    # Save the reduced dataset
    reduced_df.to_csv('data/reduced_bitcoin_data.csv', index=False)
    print("Autoencoder applied successfully. Reduced data saved to 'data/reduced_bitcoin_data.csv'")

    return reduced_df
