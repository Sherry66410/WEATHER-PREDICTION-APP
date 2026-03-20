import streamlit as st
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

st.set_page_config(
    page_title="Kerala Weather Prediction",
    page_icon="🌤️",
    layout="wide"
)

FEATURES = [
    'T2M', 'RH2M', 'WS2M', 'PRECTOTCORR', 'PS',
    'temp_lag1', 'temp_lag2', 'temp_lag3',
    'temp_roll3', 'temp_roll7',
    'month', 'day', 'day_of_week', 'week_of_year'
]

@st.cache_resource(show_spinner="Loading model...")
def load_model_artifacts():
    base = 'model_artifacts'
    if not os.path.exists(base):
        st.error(f"'{base}' folder not found!")
        return None

    try:
        scaler_X = joblib.load(f'{base}/scaler_X.pkl')
        scaler_y = joblib.load(f'{base}/scaler_y.pkl')
    except Exception as e:
        st.error(f"Error loading scalers: {e}")
        return None

    try:
        df = pd.read_csv(f'{base}/processed_weather_data.csv',
                         index_col=0, parse_dates=True)
    except Exception as e:
        st.error(f"Error loading weather data: {e}")
        return None

    # Build model and load weights from .npy files (no h5py / version issues)
    try:
        inputs  = tf.keras.Input(shape=(1, 14))
        x       = tf.keras.layers.LSTM(128)(inputs)
        x       = tf.keras.layers.Dropout(0.4)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        model   = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Trigger full build
        model(tf.zeros((1, 1, 14)))

        # Load weights from .npy files — completely version-independent
        lstm_layer  = model.layers[1]   # LSTM
        dense_layer = model.layers[3]   # Dense

        lstm_layer.set_weights([
            np.load(f'{base}/lstm_kernel.npy'),
            np.load(f'{base}/lstm_recurrent_kernel.npy'),
            np.load(f'{base}/lstm_bias.npy'),
        ])
        dense_layer.set_weights([
            np.load(f'{base}/dense_kernel.npy'),
            np.load(f'{base}/dense_bias.npy'),
        ])

    except Exception as e:
        st.error(f"Could not build model: {e}")
        return None

    return model, scaler_X, scaler_y, df


def classify_weather(temp, rain):
    if rain > 5:
        return "Rainy", "☔️", "Bring umbrella! 🌂"
    elif rain > 1:
        return "Cloudy", "☁️", "Might need a jacket 🧥"
    elif temp > 32:
        return "Hot", "☀️", "Stay hydrated! 💧"
    else:
        return "Pleasant", "😊", "Perfect day! 🌟"


def predict_tomorrow(today_input, model, scaler_X, scaler_y, df):
    try:
        new_date = df.index[-1] + pd.Timedelta(days=1)
        row = pd.DataFrame([today_input], index=[new_date])
        row['temp_lag1']    = df['T2M'].iloc[-1]
        row['temp_lag2']    = df['T2M'].iloc[-2]
        row['temp_lag3']    = df['T2M'].iloc[-3]
        row['temp_roll3']   = df['T2M'].iloc[-3:].mean()
        row['temp_roll7']   = df['T2M'].iloc[-7:].mean()
        row['month']        = new_date.month
        row['day']          = new_date.day
        row['day_of_week']  = new_date.dayofweek
        row['week_of_year'] = int(new_date.isocalendar()[1])

        X        = row[FEATURES]
        X_scaled = scaler_X.transform(X)
        X_input  = X_scaled.reshape(1, 1, X_scaled.shape[1])
        pred     = model.predict(X_input, verbose=0)
        return float(scaler_y.inverse_transform(pred)[0][0])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# Load
result = load_model_artifacts()
if result is None:
    st.stop()
best_model, scaler_X, scaler_y, df = result

# UI
st.title("🌤️ Kerala Next-Day Weather Prediction")
st.write("Enter today's weather to predict tomorrow's temperature and conditions.")

st.sidebar.header("🌡️ Today's Weather Parameters")
t2m         = st.sidebar.slider("Temperature (°C)",       10.0, 45.0,  27.0, 0.1)
rh2m        = st.sidebar.slider("Humidity (%)",            0.0, 100.0, 75.0, 0.1)
ws2m        = st.sidebar.slider("Wind Speed (m/s)",        0.0,  20.0,  2.5, 0.1)
prectotcorr = st.sidebar.slider("Precipitation (mm)",      0.0, 200.0,  0.0, 0.1)
ps_kpa      = st.sidebar.slider("Surface Pressure (kPa)", 90.0, 110.0,100.5, 0.01)

today_input = {
    "T2M": t2m, "RH2M": rh2m, "WS2M": ws2m,
    "PRECTOTCORR": prectotcorr, "PS": ps_kpa
}

st.subheader("📝 Today's Input")
st.dataframe(pd.DataFrame([{
    "Temperature (°C)": t2m,
    "Humidity (%)": rh2m,
    "Wind Speed (m/s)": ws2m,
    "Precipitation (mm)": prectotcorr,
    "Pressure (kPa)": ps_kpa
}]), use_container_width=True)

if st.button("🔮 Predict Tomorrow's Weather", type="primary"):
    with st.spinner("Calculating..."):
        temp = predict_tomorrow(today_input, best_model, scaler_X, scaler_y, df)
    if temp is not None:
        label, icon, advice = classify_weather(temp, prectotcorr)
        c1, c2, c3 = st.columns(3)
        c1.metric("🌡️ Predicted Temperature", f"{temp:.1f} °C")
        c2.metric("☁️ Weather", f"{icon} {label}")
        c3.metric("💡 Advice", advice)
        st.success("✅ Prediction complete!")

with st.expander("📈 Historical Temperature (last 90 days)"):
    st.line_chart(df['T2M'].iloc[-90:])

st.markdown("---")
st.markdown("📊 Model trained on NASA/POWER Kerala weather data using LSTM")
