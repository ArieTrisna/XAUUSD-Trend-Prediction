import streamlit as st
import pandas as pd
import joblib 
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model = joblib.load("random_forest_model.pkl")
df = pd.read_csv('./xauusd.csv')
df = df.set_index('date')
df = df.sort_index()

predictors = ['Close_Ratio2',
 'Trend_2',
 'Close_Ratio5',
 'Trend_5',
 'Close_Ratio60',
 'Trend_60',
 'Close_Ratio250',
 'Trend_250',
 'Close_Ratio1000',
 'Trend_1000']

# User input form
st.title("XAU/USD Price Direction Prediction")
multi = ''' 1 = Up  
            0 = Down'''
st.markdown(multi)

st.subheader("Here is the dataframe from 2014-01-01 to 2024-12-30")
st.dataframe(df)
st.write("Some of the data is deleted because of the NaN values from feature engineering")

st.subheader("Here is the latest date of data")
latest_date = df.tail(2)
st.dataframe(latest_date)

st.header("Click Predict to Run The Prediction Model")

# Example: input as DataFrame for prediction

if st.button("Predict"):
    prediction = model.predict(df[predictors])
    if prediction[0] == 1:
        st.success("Prediction: Price will go UP 📈")
    else:
        st.error("Prediction: Price will go DOWN 📉")
