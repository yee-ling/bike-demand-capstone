"""
Streamlit app — Bike Demand Forecaster.
========================================

CONTRACT — build a single-page Streamlit app with TWO parts:

  1. Analytics dashboard: load the cleaned data and show >= 2 EDA visualisations
     (e.g. average demand by hour, demand vs. temperature, demand by weather situation).
  2. Prediction form: collect weather + time inputs from the user, call
     ``src.predict.predict`` with the loaded champion, and display the predicted demand.

This is the file Hugging Face Spaces runs (see README front-matter: ``app_file: app.py``).
Keep it runnable with ``streamlit run app.py``.
"""
import streamlit as st
import plotly.express as px

from src.data_prep import add_features, load_and_clean
from src.predict import load_model, predict

DATA_PATH = "data/hour.csv"


def main() -> None:
    st.set_page_config(page_title="Bike Demand Forecaster", page_icon=":bike:", layout="wide")
    st.title(":bike: Bike Demand Forecaster")

    df = add_features(load_and_clean(DATA_PATH))

    st.header("Analytics Dashboard")
    col1, col2 = st.columns(2)

    with col1:
        hourly_avg = df.groupby("hr")["cnt"].mean().reset_index()
        fig1 = px.bar(hourly_avg, x="hr", y="cnt",
                      title="Average Demand by Hour")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.scatter(df, x="temp", y="cnt", opacity=0.3,
                          title="Demand vs Temperature",
                          labels={"temp": "Normalised Temperature", "cnt": "Rental Count"})
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Key Insight")
    peak_hour = int(hourly_avg.loc[hourly_avg["cnt"].idxmax(), "hr"])
    st.info(f"Demand peaks at **{peak_hour}:00** — suggesting strong commute-driven usage (morning rush).")

    st.header("Predict Hourly Demand")

    model = load_model()

    with st.form("predict_form"):
        cols = st.columns(4)
        with cols[0]:
            hr = st.slider("Hour", 0, 23, 12)
            mnth = st.selectbox("Month", range(1, 13), index=5)
        with cols[1]:
            weekday = st.selectbox("Day of Week",
                                   options=[0, 1, 2, 3, 4, 5, 6],
                                   format_func=lambda x: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][x])
            weathersit = st.selectbox("Weather Situation",
                                      options=[1, 2, 3, 4],
                                      format_func=lambda x: ["Clear", "Mist", "Light rain/snow", "Heavy rain/snow"][x - 1])
        with cols[2]:
            temp = st.slider("Temperature (normalised)", 0.0, 1.0, 0.5)
            hum = st.slider("Humidity (normalised)", 0.0, 1.0, 0.5)
        with cols[3]:
            windspeed = st.slider("Wind Speed (normalised)", 0.0, 1.0, 0.2)

        submitted = st.form_submit_button("Predict Demand", type="primary")

    if submitted:
        inputs = {
            "hr": hr,
            "mnth": mnth,
            "weekday": weekday,
            "weathersit": weathersit,
            "temp": temp,
            "hum": hum,
            "windspeed": windspeed,
        }
        prediction = predict(model, inputs)
        st.metric("Predicted Hourly Rentals", f"{prediction:,.0f}")


if __name__ == "__main__":
    main()
