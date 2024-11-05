import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import streamlit as st
import streamlit.components.v1 as components

# AdSense HTML code
adsense_html = """
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-4124087181208916"
     crossorigin="anonymous"></script>
<!-- gold-price-predictor -->
<ins class="adsbygoogle"
     style="display:block"
     data-ad-client="ca-pub-4124087181208916"
     data-ad-slot="9556206959"
     data-ad-format="auto"
     data-full-width-responsive="true"></ins>
<script>
     (adsbygoogle = window.adsbygoogle || []).push({});
</script>
"""

# Insert AdSense ad
components.html(adsense_html, height=100)


# Step 1: Read and concatenate the Excel files
files = ['Gold Prices.xlsx', 'Crude Oil Prices.xlsx', 'USDINR.xlsx', 'Silver Prices.xlsx', 'EURUSD.xlsx', 'VIX.xlsx']
dfs = [pd.read_excel(file) for file in files]

# Convert 'Date' column to datetime in each DataFrame
for df in dfs:
    df['Date'] = pd.to_datetime(df['Date'])

# Merge all dataframes on the 'Date' column
merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = merged_df.merge(df, on='Date', how='outer')

# Step 2: Preprocess the data
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df['Day'] = merged_df['Date'].dt.dayofweek  # Monday=0, Sunday=6
merged_df['Month'] = merged_df['Date'].dt.month - 1  # January=0, December=11

# Fill missing values with forward fill method
merged_df.fillna(method='ffill', inplace=True)
merged_df.fillna(method='bfill', inplace=True)

# Step 3: Prepare the data for modeling
features = merged_df.drop(['Date', 'Gold'], axis=1)
target = merged_df['Gold']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 4: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# Step 5: Create a Streamlit app
st.title('Gold Prices Prediction')

# User inputs for the features
st.sidebar.header('Input Features')
input_date = st.sidebar.date_input('Date')
input_day = input_date.weekday()
input_month = input_date.month - 1

input_features = {'Day': input_day, 'Month': input_month}
for col in features.columns:
    input_features[col] = st.sidebar.number_input(col, value=float(merged_df[col].mean()))

# Prepare the input data for prediction
input_df = pd.DataFrame([input_features])

# Ensure the order of columns in input_df matches the training data
input_df = input_df[features.columns]

# Predict the Gold Price
predicted_price = model.predict(input_df)
st.write(f'Predicted Gold Price: {predicted_price[0]}')
