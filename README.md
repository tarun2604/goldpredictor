# Gold Price Prediction using Machine Learning

## Overview
This project is a machine learning-based gold price prediction tool developed using Python. It uses a RandomForestRegressor model to predict gold prices based on various economic indicators, including crude oil prices, currency exchange rates, silver prices, and volatility index (VIX). The project is implemented using Streamlit for an interactive web-based interface.

## Features
- **Machine Learning Model**: Utilizes RandomForestRegressor for predictions.
- **Multiple Data Sources**: Uses crude oil prices, USD/INR exchange rates, silver prices, EUR/USD exchange rates, and VIX.
- **User Input Panel**: Allows users to enter custom input values for prediction.
- **Visualization**: Displays the predicted gold price interactively.
- **Google AdSense Integration**: Monetization through AdSense ads.

## Requirements
Ensure you have the following installed:
- Python 3.x
- Pandas
- Scikit-learn
- Streamlit
- OpenPyXL (for reading Excel files)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/gold-price-predictor.git
   cd gold-price-predictor
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Upload the required Excel files (`Gold Prices.xlsx`, `Crude Oil Prices.xlsx`, `USDINR.xlsx`, `Silver Prices.xlsx`, `EURUSD.xlsx`, `VIX.xlsx`).
2. The application will preprocess the data and train the model.
3. Enter the required input features in the sidebar.
4. The predicted gold price will be displayed on the main page.

## Data Processing
- **Merging**: All datasets are merged based on the date column.
- **Feature Engineering**: Extracts the day of the week and month.
- **Handling Missing Values**: Uses forward fill and backward fill methods.

## Model Training
- **Algorithm**: RandomForestRegressor with 100 estimators.
- **Train-Test Split**: 80% training, 20% testing.
- **Evaluation Metric**: Mean Absolute Error (MAE).

## Future Enhancements
- Add support for real-time data fetching.
- Improve model accuracy with advanced algorithms (e.g., XGBoost, LSTM).
- Enhance UI with better visualizations and analytics.

## License
This project is open-source and available under the MIT License.

## Contact
For any questions or suggestions, feel free to reach out or open an issue in the repository.

