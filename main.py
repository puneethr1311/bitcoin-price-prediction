import pandas as pd
import numpy as np
from data.fetch_bitcoin_data import fetch_bitcoin_data
from models.autoencoder import apply_autoencoder
from models.image_classification.cnn_model import build_and_train_cnn_model, evaluate_model
from models.image_classification.generated_candlesticks import generate_candlestick_images
from models.statistical_models.ar_model import apply_ar_model
from models.statistical_models.sarima_model import apply_sarima_model
from models.statistical_models.ma_model import apply_ma_model
from models.statistical_models.arima_model import apply_arima_model
from models.statistical_models.garch_model import apply_garch_model, grid_search_garch
from models.statistical_models.ar_garch_model import apply_ar_garch_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import PIL.Image as Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from models.lstm_model import perform_hyperparameter_tuning, train_and_evaluate_lstm
from concurrent.futures import ThreadPoolExecutor
from data_collection.twitter import fetch_twitter_data
from data_collection.reddit import fetch_reddit_data
from data_collection.news import fetch_news_data
from preprocessing.text_cleaning import preprocess_text
from models.finbert_model import FINBERTSentimentAnalyzer
from sklearn.preprocessing import MinMaxScaler
import nltk
import os
import json
import mplfinance as mpf
import matplotlib.pyplot as plt

# Set training and validation split ratio
train_ratio = 0.8

# Define all API keys
# twitter_api_key='lz0hLa6bbVMZ7itbH9sVQ0DPC'
# twitter_api_key_secret='5P2FxTrmMNxedtsWbiIyjyqbsTHqpEY8gCvW1vc5WHB1Hbt7Mr'
# twitter_bearer_token='AAAAAAAAAAAAAAAAAAAAAICPrwEAAAAAsjrJhVIkSf5Txf1QtupGRwp%2FTBI%3DybBmM02FxcVZlxk5MLeZiEBQqGYocBRrOV6IwCGs4KpHZaZxCX'

# reddit_client_id='cK3fp_KeS8JGFonzM4NOxw'
# reddit_client_secret='T9wVGIFpY5DeIIZiA1iLeySfRHZm3Q'

news_api_key='0c929d51b66f40ecacb0ef1e04f190e0'


if __name__ == "__main__":
    # Step 1: Fetch the data
    # bitcoin_data = fetch_bitcoin_data()
    bitcoin_data = pd.read_csv('data/reduced_bitcoin_data.csv', parse_dates=['Date'], index_col='Date')
    
    # # Step 2: Apply autoencoder
    # reduced_data = apply_autoencoder(bitcoin_data)

    # # Convert reduced_data to a pandas Series suitable for statistical models
    # reduced_series = pd.Series(reduced_data[:, 0], index=reduced_data.index)

    selected_features = ['Close']
    reduced_series = bitcoin_data[selected_features]

    # Split data
    train_size = int(len(reduced_series) * train_ratio)
    train_data, val_data = reduced_series[:train_size], reduced_series[train_size:]

    # Scale the data for statistical models
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
    scaled_val_data = scaler.transform(val_data.values.reshape(-1, 1)).flatten()


    # # Step 3: Apply statistical models on the entire reduced dataset(train data and test data)

    # # Step 3.1: Run AR Model
    # ar_predictions, ar_metrics = apply_ar_model(train_data, val_data, max_lags=400)
    # # ar_predictions, ar_metrics = apply_ar_model(scaled_train_data, scaled_val_data, max_lags=400)
    # # ar_predictions = scaler.inverse_transform(ar_predictions.reshape(-1, 1)).flatten()

    # # Ensure the predictions are aligned with val_data
    # aligned_index = val_data.index[:len(ar_predictions)]

    # # Define the subset size for a cleaner graph
    # subset_size = 150  # Adjust this value as needed for the number of data points
    # subset_val_data = val_data[:subset_size]
    # # subset_val_data = scaled_val_data[:subset_size]
    # subset_ar_predictions = ar_predictions[:subset_size]
    # subset_index = aligned_index[:subset_size]

    # # Plot the results
    # plt.figure(figsize=(12, 6))
    # plt.plot(subset_index, subset_val_data, label="Actual Values", color="#FFA500", linewidth=2, linestyle='solid')
    # plt.plot(subset_index, subset_ar_predictions, label="AR Predicted Values", color="#05ED98", linewidth=2, linestyle='solid')
    # plt.title("Autoregressive Model (AR) - Prediction vs. Actual", fontsize=14, fontweight="bold", color="#333333")
    # plt.legend(fontsize=12, frameon=True, facecolor="#FFFFFF", edgecolor="#000000")
    # plt.xticks(fontsize=10, fontweight="600", color="#333333")
    # plt.yticks(fontsize=10, fontweight="600", color="#333333")
    # plt.xlabel("Time", fontsize=14)
    # plt.ylabel("Price", fontsize=14)
    # plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    # plt.show()
    
    # print("AR Model Metrics:", ar_metrics)



    # Step 3.2: Run GARCH Model
    # # Perform grid search to find the best (p, q) values
    # p_values = range(1, 10)  # Searching within a reasonable range
    # q_values = range(1, 10)

    # best_garch_predictions, best_garch_metrics, best_garch_params = grid_search_garch(
    #     scaled_train_data, scaled_val_data, p_values, q_values
    # )

    # print(f"Optimal GARCH Parameters: p={best_garch_params[0]}, q={best_garch_params[1]}")

    # # Apply GARCH with the best (p, q) values
    # garch_predictions, garch_metrics = apply_garch_model(
    #     scaled_train_data, scaled_val_data, p=4, q=2 
    # )
    # print(f"Predictions Length: {len(garch_predictions)}, Validation Data Length: {len(val_data.index)}")

    # # Inverse transform predictions and validation data to original scale
    # # garch_predictions_original_scale = scaler.inverse_transform(garch_predictions.reshape(-1, 1)).flatten()
    # # val_data_original_scale = scaler.inverse_transform(val_data.values.reshape(-1, 1)).flatten()

    # # Ensure predictions align with validation data index
    # aligned_index = val_data.index[:len(garch_predictions)]

    # # Define the subset size for a cleaner graph
    # subset_size = 280  # Adjust this value as needed for the number of data points
    # subset_val_data = scaled_val_data[:subset_size]
    # subset_garch_predictions = garch_predictions[:subset_size]
    # # subset_garch_predictions_original_scale = garch_predictions_original_scale[:subset_size]
    # subset_index = aligned_index[:subset_size]

    # # Plot the results
    # plt.figure(figsize=(12, 6))
    # plt.plot(subset_index, subset_val_data, label="Actual Values", color="#FF5733", linewidth=2, linestyle='solid')
    # plt.plot(subset_index, subset_garch_predictions, label="GARCH Modeled Volatility", color="#0818A8", linewidth=1.5, linestyle='solid')
    # plt.title("GARCH Model - Volatility Estimation", fontsize=14, fontweight='bold', color="#333333")
    # plt.legend(fontsize=12, frameon=True, facecolor="#FFFFFF", edgecolor="#000000")
    # plt.xticks(fontsize=10, fontweight='600', color="#333333")
    # plt.yticks(fontsize=10, fontweight='600', color="#333333")
    # plt.xlabel("Time", fontsize=14)
    # plt.ylabel("Volatility", fontsize=14)
    # plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    # plt.show()

    # print("GARCH Model - Estimated Volatility Metrics:", garch_metrics)



    # # Apply the AR-GARCH model
    # ar_lags = 400  # Using 400 lags as it performed best in AR model
    # garch_p = 4
    # garch_q = 2

    # ar_garch_predictions, ar_garch_metrics = apply_ar_garch_model(scaled_train_data, scaled_val_data, ar_lags, garch_p, garch_q, original_index=bitcoin_data.index)

    # # Inverse transform predictions and validation data to match the original scale
    # ar_garch_predictions_original_scale = scaler.inverse_transform(ar_garch_predictions.values.reshape(-1, 1)).flatten()
    # # val_data_original_scale = scaler.inverse_transform(val_data.values.reshape(-1, 1)).flatten()

    # # Ensure the predictions are aligned with validation data index
    # aligned_index = val_data.index

    # # Define the subset size for a cleaner graph
    # subset_size = 150  # Adjust this value as needed for the number of data points
    # subset_val_data = scaled_val_data[:subset_size]
    # # subset_ar_garch_predictions_original_scale = ar_garch_predictions_original_scale[:subset_size]
    # subset_ar_garch_predictions = ar_garch_predictions[:subset_size]
    # subset_index = aligned_index[:subset_size]

    # # Plot the results
    # plt.figure(figsize=(12, 6))
    # plt.plot(subset_index, subset_val_data, label="Actual Values", color="#000080", linewidth=2, linestyle='solid')
    # plt.plot(subset_index, subset_ar_garch_predictions, label="AR-GARCH Predicted Values", color="#05ED98", linewidth=2, linestyle='solid')
    # # plt.plot(subset_index, subset_ar_garch_predictions_original_scale, label="AR-GARCH Predicted Values", color="#05ED98", linewidth=2, linestyle='solid')
    # plt.title("AR-GARCH Model - Prediction vs. Actual", fontsize=14, fontweight='bold', color="#333333")
    # plt.legend(fontsize=12, frameon=True, facecolor="#FFFFFF", edgecolor="#000000")
    # plt.xticks(fontsize=10, fontweight='600', color="#333333")
    # plt.yticks(fontsize=10, fontweight='600', color="#333333")
    # plt.xlabel("Time", fontsize=14)
    # plt.ylabel("Normalized Price Value", fontsize=14)
    # plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    # plt.show()

    # # Print evaluation metrics
    # print("AR-GARCH Model Metrics:", ar_garch_metrics)






    # # Step 4: Integrate Candlestick Charts Pattern Analysis

    # # Step 4.1: Generate Candlestick Images for validation period
    # print("Columns in bitcoin_data:", bitcoin_data.columns)
    # # generate_candlestick_images(bitcoin_data)

    # # Align predictions to the same length
    # min_length = len(ar_garch_predictions)
    # ar_garch_pred = ar_garch_predictions[:min_length]
    # val_data_aligned = val_data[-min_length:]

    # # Load the CNN model and the pre-existing labels file
    # cnn_model = load_model('models/image_classification/candlestick_cnn_model.h5')
    # labels_df = pd.read_csv('data/candlestick_images/candlestick_labels.csv')

    # pattern_adjustments = []
    # pattern_labels = []

    # window = 30  # Ensure this matches the window size used in generate_candlestick_images
    # expected_image_size = (128, 128)  # Matches the input size of the CNN model

    # for i, row in labels_df.iterrows():
    #     if i >= min_length:  # Limit processing to `min_length`
    #         break
    #     img_path, label = row['file_path'], row['label']
    #     img = Image.open(img_path)
    
    #     # Convert RGBA to RGB if needed
    #     if img.mode == 'RGBA':
    #         img = img.convert('RGB')
    
    #     # Resize and normalize the image
    #     img = img.resize(expected_image_size)  # Ensure the size matches the model input
    #     img = np.array(img) / 255.0   # Normalize pixel values
    #     img = np.expand_dims(img, axis=0)  # Add batch dimension

    #     # Debugging check
    #     print(f"Image shape before prediction: {img.shape}")

    #     # Predict label using the CNN model
    #     pattern_prediction = cnn_model.predict(img)[0][0]
    #     pred_label = 'bullish' if pattern_prediction > 0.5 else 'bearish'
    #     pattern_labels.append(pred_label)

    #     adjustment = 1.01 if pred_label == 'bullish' else 0.99
    #     pattern_adjustments.append(adjustment)

    # # Pad pattern_adjustments if necessary
    # if len(pattern_adjustments) < min_length:
    #     pattern_adjustments.extend([1.0] * (min_length - len(pattern_adjustments)))

    # # Compute the ensemble predictions
    # ensemble_predictions = ar_garch_pred * np.array(pattern_adjustments[:min_length])

    # # Inverse transform predictions and validation data to match the original scale
    # ensemble_predictions_original_scale = scaler.inverse_transform(ensemble_predictions.values.reshape(-1, 1)).flatten()

    # # Step 5: Calculate Metrics for Ensemble Model
    # mae = mean_absolute_error(val_data[-min_length:], ensemble_predictions)
    # mse = mean_squared_error(val_data[-min_length:], ensemble_predictions)
    # rmse = np.sqrt(mse)
    # ensemble_metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
    # print("Ensemble Model Metrics:", ensemble_metrics)

    # # Align predictions with validation data
    # aligned_index = val_data.index

    # # Define the subset size for a cleaner graph
    # subset_size = 185  # Adjust this value as needed for the number of data points
    # subset_val_data = scaled_val_data[:subset_size]
    # subset_ensemble_predictions = ensemble_predictions[:subset_size]
    # # subset_ensemble_predictions_original_scale = ensemble_predictions_original_scale[:subset_size]
    # subset_index = aligned_index[:subset_size]

    # # Plot the ensemble results
    # plt.figure(figsize=(12, 6))
    # plt.plot(subset_index, subset_val_data, label="Actual Values", color="#0818A8", linewidth=1.5, linestyle="solid")
    # plt.plot(subset_index, subset_ensemble_predictions, label="Ensemble Predicted Values", color="#32CD32", linewidth=1.5, linestyle="solid")
    # # plt.plot(subset_index, subset_ensemble_predictions_original_scale, label="Ensemble Predicted Values", color="#05ED98", linewidth=2, linestyle="solid")
    # plt.title("Ensemble Model (AR-GARCH + Candlestick Analysis) - Prediction vs. Actual", fontsize=14, fontweight="bold", color="#333333")
    # plt.legend(fontsize=12, frameon=True, facecolor="#FFFFFF", edgecolor="#000000")
    # plt.xticks(fontsize=10, fontweight="600", color="#333333")
    # plt.yticks(fontsize=10, fontweight="600", color="#333333")
    # plt.xlabel("Time", fontsize=14)
    # plt.ylabel("Normalized Price Value", fontsize=14)
    # plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    # plt.show()
    


    # # Step 6: LSTM Model Implementation
    # print("Applying Long Short Term Memory (LSTM) Model...")

    # # Define hyperparameter ranges
    # look_back_values = [15, 30, 60]
    # epochs_values = [200]
    # batch_size_values = [16, 32]
    # lstm_units_values = [50, 100]
    # dropout_rate_values = [0.2, 0.3]

    # # Perform hyperparameter tuning
    # best_config, best_metrics = perform_hyperparameter_tuning(
    #     reduced_series.values,
    #     look_back_values,
    #     epochs_values,
    #     batch_size_values,
    #     lstm_units_values,
    #     dropout_rate_values
    # )

    # # Apply the best configuration
    # best_look_back = best_config["look_back"]
    # lstm_model, lstm_metrics, scaler, lstm_predictions, actual_values = train_and_evaluate_lstm(
    #     reduced_series.values,
    #     look_back=best_look_back,
    #     epochs=best_config["epochs"],
    #     batch_size=best_config["batch_size"],
    #     lstm_units=best_config["lstm_units"],
    #     dropout_rate=best_config["dropout_rate"]
    # )

    # # Ensure the predictions are aligned with validation data index
    # aligned_index = val_data.index[-len(lstm_predictions):]

    # # Plot results for all models
    # plt.figure(figsize=(12, 6))
    # plt.plot(aligned_index, actual_values, label='Actual Values', color='orange')
    # plt.plot(aligned_index, lstm_predictions, label='LSTM Predicted Values', color='cyan')
    # plt.title("LSTM Model - Prediction vs. Actual")
    # plt.legend()
    # plt.show()

    # print("LSTM Model Metrics:", lstm_metrics)


    # # Step 6: LSTM Model Implementation
    # print("Applying Long Short Term Memory (LSTM) Model...")

    # # Using the best configuration from the hyperparameter tuning
    # best_look_back = 30
    # best_epochs = 200
    # best_batch_size = 16
    # best_lstm_units = 100
    # best_dropout_rate = 0.2

    # lstm_model, lstm_metrics, scaler, lstm_predictions, actual_values = train_and_evaluate_lstm( 
    #     reduced_series.values,
    #     look_back=best_look_back,
    #     epochs=best_epochs,
    #     batch_size=best_batch_size,
    #     lstm_units=best_lstm_units,
    #     dropout_rate=best_dropout_rate
    # )

    # # Ensure the predictions are aligned with validation data index
    # train_ratio = 0.8
    # train_size = int(len(reduced_series) * train_ratio)
    # val_data = reduced_series[train_size:]
    # aligned_index = val_data.index[-len(lstm_predictions):]

    # # Plot the results
    # plt.figure(figsize=(12, 6))
    # plt.plot(aligned_index, actual_values, label='Actual Values', color='orange', linewidth=1, linestyle="solid")
    # plt.plot(aligned_index, lstm_predictions, label='LSTM Predicted Values', color='cyan', linewidth=1, linestyle="solid")
    # plt.title("LSTM Model - Prediction vs. Actual", fontsize=14, fontweight="bold", color="#333333")
    # plt.legend(fontsize=12, frameon=True, facecolor="#FFFFFF", edgecolor="#000000")
    # plt.xticks(fontsize=10, fontweight="600", color="#333333")
    # plt.yticks(fontsize=10, fontweight="600", color="#333333")
    # plt.xlabel("Time", fontsize=14)
    # plt.ylabel("Price", fontsize=14)
    # plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    # plt.show()

    # print("LSTM Model Metrics:", lstm_metrics)
    


    # # Step 7: Combine Results from All Models
    # print("Combining results from AR, AR-GARCH, LSTM, and Candlestick Models...")

    # # Ensure all predictions are aligned to the same length
    # min_length = min(
    #     len(lstm_predictions),
    #     len(ensemble_predictions),
    #     len(val_data)  # Match to validation data length
    # )

    # # Align predictions and validation data
    # lstm_pred = lstm_predictions[:min_length]
    # ensemble_pred = ensemble_predictions[:min_length]
    # val_data_aligned = val_data[-min_length:]  # Match to validation data length

    # # Assign weights to each model
    # weights = {
    #     "LSTM": 0.75, 
    #     "Ensemble": 0.25,
    # }

    # # Calculate weighted ensemble prediction
    # combined_predictions = (
    #     weights["LSTM"] * lstm_pred +
    #     weights["Ensemble"] * ensemble_pred
    # )

    # # Inverse transform predictions and validation data to match the original scale
    # val_data_original_scale = scaler.inverse_transform(val_data_aligned.to_numpy().reshape(-1, 1)).flatten()
    # combined_predictions_original_scale = scaler.inverse_transform(combined_predictions.to_numpy().reshape(-1, 1)).flatten()

    # # Evaluate combined predictions
    # # combined_mae = mean_absolute_error(val_data_aligned, combined_predictions)
    # # combined_mse = mean_squared_error(val_data_aligned, combined_predictions)
    # combined_mae = mean_absolute_error(val_data_original_scale, combined_predictions_original_scale)
    # combined_mse = mean_squared_error(val_data_original_scale, combined_predictions_original_scale)
    # combined_rmse = np.sqrt(combined_mse)
    # combined_metrics = {
    #     "MAE": combined_mae,
    #     "MSE": combined_mse,
    #     "RMSE": combined_rmse
    # }

    # # Print combined metrics
    # print("Combined Model Metrics:", combined_metrics)

    # # Align the index for the plot
    # aligned_index = val_data.index[-min_length:]

    # # Define the subset size for a cleaner graph
    # subset_size = 175  # Adjust this value as needed for the number of data points
    # subset_val_data = val_data_original_scale[:subset_size]
    # # subset_combined_predictions = combined_predictions[:subset_size]
    # subset_combined_predictions_original_scale = combined_predictions_original_scale[:subset_size]
    # subset_index = aligned_index[:subset_size]

    # # Plot combined results
    # plt.figure(figsize=(12, 6))
    # plt.plot(subset_index, subset_val_data, label="Actual Values", color="#E67E22", linewidth=2, linestyle="solid")
    # plt.plot(subset_index, subset_combined_predictions_original_scale, label="Combined Predicted Values", color="cyan", linewidth=2, linestyle="solid")
    # # plt.plot(subset_index, subset_combined_predictions, label="Combined Predicted Values", color="#0437F2", linewidth=2, linestyle="solid")
    # plt.title("Combined Model (AR + AR-GARCH + LSTM + Candlestick) - Prediction vs. Actual", fontsize=14, fontweight="bold", color="#333333")
    # plt.legend(fontsize=12, frameon=True, facecolor="#FFFFFF", edgecolor="#000000")
    # plt.xticks(fontsize=10, fontweight="600", color="#333333")
    # plt.yticks(fontsize=10, fontweight="600", color="#333333")
    # plt.xlabel("Time", fontsize=14)
    # plt.ylabel("Normalized Price Value", fontsize=14)
    # plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    # plt.show()



    # # Step 8: Sentiment Analysis Implementation using FINBERT
    # print("Sentiment Analysis with FINBERT...")

    # # Load datasets
    # print("Loading datasets...")

    # # Load Bitcoin tweets dataset
    # btc_tweets_path = r"c:\Users\md mursaleen\Downloads\BTC_Tweets.csv"
    # btc_tweets_data = pd.read_csv(btc_tweets_path)
    # btc_tweets_texts = btc_tweets_data['Tweet'].dropna().tolist()

    # # Load Reddit Bitcoin dataset
    # reddit_data_path = r"c:\Users\md mursaleen\Downloads\reddit_bitcoin.csv"
    # reddit_data = pd.read_csv(reddit_data_path)
    # reddit_texts = reddit_data['body'].dropna().tolist()

    # # Fetch news data
    # print("Fetching news data...")
    # news_texts = fetch_news_data(news_api_key, query="Bitcoin", max_results=100)

    # # # API keys
    # # twitter_credentials = {
    # #     "api_key": twitter_api_key,
    # #     "api_key_secret": twitter_api_key_secret,
    # #     "bearer_token": twitter_bearer_token
    # # }

    # # reddit_credentials = {
    # #     "client_id": reddit_client_id,
    # #     "client_secret": reddit_client_secret,
    # #     "user_agent": "Coin Tracking by AdSecret2106",
    # # }

    # # news_api_key = news_api_key

    # # # Fetch social media data in parallel
    # # print("Fetching data...")
    # # with ThreadPoolExecutor() as executor:
    # #     futures = [
    # #         executor.submit(fetch_twitter_data, **twitter_credentials, query="Bitcoin", max_tweets=100),
    # #         executor.submit(fetch_reddit_data, **reddit_credentials, subreddit_name="Bitcoin", query="Bitcoin", max_posts=100),
    # #         executor.submit(fetch_news_data, news_api_key, query="Bitcoin", max_results=100)
    # #     ]
    # #     twitter_data, reddit_data, news_data = [future.result() for future in futures]

    # # Combining all the data
    # all_texts = btc_tweets_texts + reddit_texts + news_texts
    # all_sources = (["Twitter"] * len(btc_tweets_texts)) + (["Reddit"] * len(reddit_texts)) + (["News"] * len(news_texts))
    # assert len(all_texts) == len(all_sources), "Mismatch between text data and sources."

    # print(f"Total data points collected: {len(all_texts)}")

    # # Preprocess text
    # print("Preprocessing text...")
    # preprocessed_texts_file = "preprocessed_texts.json"

    # if os.path.exists(preprocessed_texts_file):
    #     print("Loading preprocessed texts from file...")
    #     with open(preprocessed_texts_file, "r") as f:
    #         preprocessed_texts = json.load(f)
    # else:
    #     with ThreadPoolExecutor() as executor:
    #         preprocessed_texts = list(executor.map(preprocess_text, all_texts))
    #     with open(preprocessed_texts_file, "w") as f:
    #         json.dump(preprocessed_texts, f)

    # # Save preprocessed texts to file
    # with open(preprocessed_texts_file, "w") as f:
    #     json.dump(preprocessed_texts, f)
    # print(f"Preprocessed texts saved to {preprocessed_texts_file}")

    # # Filter out empty texts and synchronize sources
    # valid_indices = [i for i, text in enumerate(preprocessed_texts) if text.strip()]
    # preprocessed_texts = [preprocessed_texts[i] for i in valid_indices]
    # all_sources = [all_sources[i] for i in valid_indices]

    # print(f"Valid preprocessed texts: {len(preprocessed_texts)}")
    # print(f"Valid sources: {len(all_sources)}")
    # assert len(preprocessed_texts) == len(all_sources), "Mismatch between filtered texts and sources."  

    # # Sentiment analysis
    # print("Performing sentiment analysis...")
    # sentiment_scores_file = "sentiment_scores.json"
    # sentiment_analyzer = FINBERTSentimentAnalyzer()  # Ensure this is initialized before using

    # if os.path.exists(sentiment_scores_file):
    #     print("Loading precomputed sentiment scores...")
    #     with open(sentiment_scores_file, "r") as f:
    #         sentiment_scores = json.load(f)
    # else:
    #     batch_size = 500
    #     sentiment_scores = []
    #     for i in range(0, len(preprocessed_texts), batch_size):
    #         batch_texts = preprocessed_texts[i:i + batch_size]
    #         print(f"Processing batch {i // batch_size + 1} with {len(batch_texts)} texts...")
    #         try:
    #             batch_scores = sentiment_analyzer.batch_analyze_sentiment(batch_texts)
    #             sentiment_scores.extend(batch_scores.to_dict(orient='records'))  # Convert DataFrame to list of dicts
    #         except Exception as e:
    #             print(f"Error processing batch {i // batch_size + 1}: {e}")
    #             # Fill with default values to maintain alignment
    #             sentiment_scores.extend([{"positive": 0, "neutral": 0, "negative": 0}] * len(batch_texts))

    # print(f"Total sentiment scores: {len(sentiment_scores)}")

    # # Save sentiment scores to file
    # with open(sentiment_scores_file, "w") as f:
    #     json.dump(sentiment_scores, f)
    # print(f"Sentiment scores saved to {sentiment_scores_file}")

    # # Ensure the number of sentiment scores matches the sources
    # assert len(sentiment_scores) == len(all_sources), f"Mismatch: {len(sentiment_scores)} sentiment scores vs. {len(all_sources)} sources"

    # # Create a DataFrame
    # print("Creating results DataFrame...")
    # sentiment_results = pd.DataFrame(sentiment_scores)
    # sentiment_results["source"] = all_sources
    # sentiment_results["date"] = pd.Timestamp.now().normalize()

    # # Convert numeric columns
    # numeric_columns = ["positive", "neutral", "negative"]
    # sentiment_results[numeric_columns] = sentiment_results[numeric_columns].apply(pd.to_numeric, errors="coerce")
    # sentiment_results[numeric_columns] = sentiment_results[numeric_columns].fillna(0)

    # # Save results
    # sentiment_results.to_csv("sentiment_results.csv", index=False)
    # print("Sentiment analysis results saved to 'sentiment_results.csv'.")

    # # Debug: Print the first few rows of the DataFrame
    # print(sentiment_results.head())

    # # Plot results
    # print("Plotting sentiment trends...")
    # sentiment_results["date"] = pd.to_datetime(sentiment_results["date"])
    # sentiment_results.set_index("date", inplace=True)

    # # Handle case where all data points are on the same date
    # if sentiment_results.index.nunique() == 1:
    #     print("All data points are from the same date. Adding synthetic time intervals...")
    #     sentiment_results = sentiment_results.reset_index()
    #     sentiment_results["date"] = pd.date_range(start="2024-12-01", periods=len(sentiment_results), freq="h")
    #     sentiment_results.set_index("date", inplace=True)

    # # Plot only numeric sentiment columns
    # sentiment_trends = sentiment_results[numeric_columns].resample("D").mean()
    # start_date = sentiment_results.index.min().strftime("%Y-%m-%d")
    # end_date = sentiment_results.index.max().strftime("%Y-%m-%d")

    # plt.figure(figsize=(12, 6))
    # plt.plot(sentiment_trends.index, sentiment_trends["positive"], label="Positive", color="#2ECC71", linewidth=2)
    # plt.plot(sentiment_trends.index, sentiment_trends["neutral"], label="Neutral", color="#F1C40F", linewidth=2)
    # plt.plot(sentiment_trends.index, sentiment_trends["negative"], label="Negative", color="#E74C3C", linewidth=2)
    # plt.title(f"Sentiment Trends Over Time ({start_date} to {end_date})", fontsize=14, fontweight="bold", color="#333333")
    # plt.xlabel("Date", fontsize=12)
    # plt.ylabel("Average Sentiment Scores", fontsize=12)
    # plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    # plt.legend(fontsize=12, frameon=True, loc="upper right")
    # plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    # plt.show()



    # # Step 9: Combine Ensemble Model and Sentiment Analysis
    # print("Step 9: Combining Ensemble Model and Sentiment Analysis...")

    # # Align ensemble predictions and validation data
    # min_length = min(
    #     len(ar_predictions),
    #     len(garch_predictions),
    #     len(ar_garch_predictions),
    #     len(lstm_predictions),
    #     len(ensemble_adjusted),
    #     len(val_data),  # Align with validation data length
    #     len(sentiment_results)  # Match with sentiment results length
    # )

    # # Align predictions and validation data
    # ar_pred = ar_predictions[:min_length]
    # garch_pred = garch_predictions[:min_length]
    # ar_garch_pred = ar_garch_predictions[:min_length]
    # lstm_pred = lstm_predictions[:min_length]
    # candlestick_adjusted_pred = ensemble_adjusted[:min_length]
    # sentiment_scores = sentiment_results[["positive", "neutral", "negative"]].iloc[:min_length]
    # actual_vals = val_data.values[-min_length:]  # Match to validation data length

    # # Align sentiment results with predictions
    # sentiment_results_aligned = sentiment_results.iloc[-min_length:]
    # sentiment_positive = sentiment_results_aligned["positive"].values
    # sentiment_neutral = sentiment_results_aligned["neutral"].values
    # sentiment_negative = sentiment_results_aligned["negative"].values

    # # Assign weights to each model and sentiment scores
    # weights = {
    #     "AR": 0.15,
    #     "GARCH": 0.10,
    #     "AR-GARCH": 0.10,
    #     "LSTM": 0.40,
    #     "Candlestick": 0.15,
    #     "Sentiment": 0.1,
    # }

    # # Calculate the final combined prediction
    # final_combined_predictions = (
    #     weights["AR"] * ar_pred +
    #     weights["GARCH"] * garch_pred +
    #     weights["AR-GARCH"] * ar_garch_pred +
    #     weights["LSTM"] * lstm_pred +
    #     weights["Candlestick"] * candlestick_adjusted_pred +
    #     weights["Sentiment"] * (sentiment_scores["positive"].values - sentiment_scores["negative"].values)
    # )

    # # Inverse transform predictions and validation data to match the original scale
    # final_combined_predictions_original_scale = scaler.inverse_transform(final_combined_predictions.values.reshape(-1, 1)).flatten()
    # actual_values_rescaled = scaler.inverse_transform(actual_vals.reshape(-1, 1)).flatten()

    # # Evaluate final combined model
    # final_mae = mean_absolute_error(actual_values_rescaled, final_combined_predictions_original_scale)
    # final_mse = mean_squared_error(actual_values_rescaled, final_combined_predictions_original_scale)
    # final_rmse = np.sqrt(final_mse)
    # final_metrics = {
    #     "MAE": final_mae,
    #     "MSE": final_mse,
    #     "RMSE": final_rmse
    # }
  
    # # Print final combined model metrics
    # print("Final Combined Model Metrics:", final_metrics)

    # # Align the index for the plot
    # aligned_index = val_data.index

    # # Define the subset size for a cleaner graph
    # subset_size = 200  # Adjust this value as needed for the number of data points
    # subset_val_data = actual_values_rescaled[:subset_size]
    # subset_final_combined_predictions_original_scale = final_combined_predictions_original_scale[:subset_size]
    # subset_index = aligned_index[:subset_size]

    # # Plot the final combined results (Default Background)
    # plt.figure(figsize=(12, 6))
    # plt.plot(subset_index, subset_val_data, label="Actual Values", color="orange")
    # plt.plot(subset_index, subset_final_combined_predictions_original_scale, label="Final Predicted Values", color="cyan")
    # plt.title("Final Combined Model - Prediction vs. Actual")
    # plt.legend()
    # plt.xlabel("Time")
    # plt.ylabel("Price")
    # plt.grid(True)
    # plt.show()

    # # Plot the final combined results (Black Background)
    # plt.figure(figsize=(12, 6))
    # plt.plot(subset_index, subset_val_data, label="Actual Values", color="white")
    # plt.plot(subset_index, subset_final_combined_predictions_original_scale, label="Final Predicted Values", color="cyan")
    # plt.title("Final Combined Model - Prediction vs. Actual (Black Background)", color="white")
    # plt.legend(facecolor="black", edgecolor="white")
    # plt.xlabel("Time", color="white")
    # plt.ylabel("Price", color="white")
    # plt.grid(color="gray")
    # plt.gca().set_facecolor("black")  # Set the plot background to black
    # plt.gcf().set_facecolor("black")  # Set the figure background to black
    # plt.xticks(color="white")
    # plt.yticks(color="white")
    # plt.show()



    # Candlestick Image Analysis

    # Step 1: Load historical data
    bitcoin_data = pd.read_csv('data/bitcoin_data.csv', parse_dates=['Date'], index_col='Date')

    # Convert timezone-aware datetime to naive datetime if necessary
    if bitcoin_data.index.tzinfo is not None:
        print("Converting timezone-aware datetime index to naive datetime...")
        bitcoin_data.index = bitcoin_data.index.tz_convert(None)

    # Ensure the required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for column in required_columns:
        if column not in bitcoin_data.columns:
            print(f"Column '{column}' not found in the dataset. Adding placeholder values.")
            bitcoin_data[column] = 0  # Add placeholder column if missing

    # Step 2: Generate candlestick images for pattern recognition
    generate_candlestick_images(bitcoin_data)

    # Step 3: Train CNN model to recognize bullish/bearish patterns
    model, history = build_and_train_cnn_model()

    # Step 4: Evaluate CNN model
    test_loss, test_accuracy = evaluate_model(test_dir='data/candlestick_images')
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Load candlestick labels
    labels_path = 'data/candlestick_images/candlestick_labels.csv'
    labels_df = pd.read_csv(labels_path)

    # Load the trained CNN model
    model_path = 'models/image_classification/candlestick_cnn_model.h5'
    cnn_model = load_model(model_path)

    # Prepare data for plotting
    window = 30  # Window size used during candlestick image generation
    image_size = (128, 128)  # Image size used during training

    predictions = []
    for i, row in labels_df.iterrows():
        img_path = row['file_path']
        img = Image.open(img_path).convert('RGB')
        img = img.resize(image_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = cnn_model.predict(img_array)[0][0]
        predictions.append('bullish' if prediction > 0.5 else 'bearish')

    # Add predictions to the labels DataFrame
    labels_df['predicted_label'] = predictions

    # Annotate the candlestick chart with predictions
    candlestick_data = bitcoin_data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[:len(labels_df) * window]
    predicted_adjustments = []

    for i in range(len(labels_df)):
        label = labels_df.loc[i, 'predicted_label']
        adjustment = 1.01 if label == 'bullish' else 0.99
        predicted_adjustments.extend([adjustment] * window)

    candlestick_data['Predicted Close'] = candlestick_data['Close'] * predicted_adjustments[:len(candlestick_data)]

    # Ensure 'candlestick_data' index is datetime and contains no invalid entries
    if not isinstance(candlestick_data.index, pd.DatetimeIndex):
        print("Index is not a DatetimeIndex. Attempting to convert...")
        candlestick_data.index = pd.to_datetime(candlestick_data.index, errors='coerce')

    # Drop rows with invalid datetime index
    if candlestick_data.index.isnull().any():
        print("Dropping rows with invalid datetime index...")
        candlestick_data = candlestick_data[~candlestick_data.index.isnull()]

    # Ensure all required columns are numeric
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        if candlestick_data[col].dtype != 'float64' and candlestick_data[col].dtype != 'int64':
            print(f"Converting column '{col}' to numeric...")
            candlestick_data[col] = pd.to_numeric(candlestick_data[col], errors='coerce')

    # Drop rows with missing or NaN values
    candlestick_data = candlestick_data.dropna(subset=numeric_columns)

    # Select a reasonable subset of data for better visualization
    subset_data = candlestick_data[-150:]  # Last 150 rows of data for plotting

    # Debugging subset to ensure it is correct
    print(subset_data.head())
    print(subset_data.info())

    # Create figure and axis manually
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot candlestick chart for the subset
    custom_style = mpf.make_mpf_style(
        base_mpf_style='charles',
        facecolor='#FFFFFF',
        edgecolor='#000000',
        gridcolor='#808080',
        rc={
            'axes.labelcolor': '#000000',
            'xtick.color': '#000000',
            'ytick.color': '#000000'
        }
    )

    apd = [mpf.make_addplot(subset_data['Predicted Close'], color='#3F00FF', linestyle='dashed', width=1.5, label="Predicted Close")]

    mpf.plot(
        subset_data,
        type='candle',
        style=custom_style,
        addplot=apd,
        title='Candlestick Chart: Predicted vs Actual (Subset)',
        ylabel='Price',
        ylabel_lower='Volume',
        volume=True,
        datetime_format='%b %d',  # Format x-axis labels (e.g., "Jan 15")
        xrotation=0,  # Keep x-axis labels straight
        # fig=fig,  # Pass the figure
        # ax=ax  # Pass the axis
        figscale=1.2
    )
    
    
    
    
    