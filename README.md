# Bitcoin Price Prediction and Analysis

## Overview
This project analyzes Bitcoin price trends using various machine learning and statistical models. It includes deep learning-based autoencoders, convolutional neural networks (CNNs) for image-based analysis, and traditional time series forecasting models like AR, SARIMA, and MA.

## Features
- **Data Fetching**: Retrieves Bitcoin price data.
- **Autoencoder**: Applies deep learning for anomaly detection.
- **CNN Model**: Trains and evaluates a CNN for candlestick image classification.
- **Statistical Models**: Uses AR, SARIMA, and MA models for time series forecasting.

## Installation
Ensure you have Python installed, then install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the `main.py` script to execute the full pipeline:
```bash
python main.py
```

## Directory Structure
```
project-root/
│── main.py
│── data/
│   ├── fetch_bitcoin_data.py
│── models/
│   ├── autoencoder.py
│   ├── statistical_models/
│   │   ├── ar_model.py
│   │   ├── sarima_model.py
│   │   ├── ma_model.py
│   ├── image_classification/
│   │   ├── cnn_model.py
│   │   ├── generated_candlesticks.py
│── README.md
│── requirements.txt
```

## Contributing
Feel free to submit pull requests or open issues for improvements.

## License
This project is licensed under the MIT License.

