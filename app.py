import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from models.lstm_forecast import train_lstm_model, forecast_future

st.set_page_config(page_title="📈 Time Series Forecasting Dashboard", layout="wide")

st.title("📊 Time Series Forecasting Dashboard (LSTM)")
st.write("Upload your time series data (CSV) to forecast future trends using an LSTM model.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.write(df.head())

    col = st.selectbox("Select the column to forecast", df.columns)
    seq_length = st.slider("Sequence Length", 5, 50, 10)
    epochs = st.slider("Training Epochs", 5, 100, 20)
    steps = st.slider("Forecast Steps", 5, 50, 10)

    if st.button("🚀 Train & Forecast"):
        with st.spinner("Training model... please wait ⏳"):
            model, scaler = train_lstm_model(df[col], seq_length, epochs)
            predictions = forecast_future(model, scaler, df[col].values, seq_length, steps)

        st.success("✅ Training complete!")
        forecast_index = range(len(df), len(df) + steps)
        df_forecast = pd.DataFrame({"Index": forecast_index, "Forecast": predictions})

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df[col], label="Original Data", color="blue")
        ax.plot(forecast_index, predictions, label="Forecast", color="orange")
        ax.legend()
        ax.set_title("Time Series Forecast")
        st.pyplot(fig)

        st.subheader("🔮 Forecasted Values")
        st.write(df_forecast)
else:
    st.info("👆 Please upload a CSV file to begin.")
