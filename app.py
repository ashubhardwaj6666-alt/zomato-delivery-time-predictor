"""
Zomato Delivery Time Predictor
================================
A Streamlit front-end for a Linear Regression model trained to predict
food delivery time from order, delivery-partner, weather, traffic and
location features.

The prediction pipeline below intentionally mirrors, step for step, the
preprocessing performed in the training notebook (feature engineering,
outlier capping, ordinal/one-hot encoding, scaling) so predictions made
here stay consistent with what the model actually learned.

Form design note: only fields a real customer would actually know
(order type, city, distance) are asked up front. Fields that belong to
the system in a live product (assigned delivery partner, live weather/
traffic, current time) are defaulted sensibly and tucked into an
"advanced" expander for testing.
"""

from __future__ import annotations

import base64
from datetime import datetime, date, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
BRAND_RED = "#CB202D"
LOGO_PATH = Path("zomato_logo.png")  # drop your own logo file here if you have one

ARTIFACT_PATHS = {
    "model": "zomato_model.pkl",
    "scaler": "zomato_scaler.pkl",
    "feature_columns": "zomato_feature_columns.pkl",
    "iqr_bounds": "zomato_iqr_bounds.pkl",
    "category_options": "zomato_category_options.pkl",
    "metrics": "zomato_metrics.pkl",
}

TRAFFIC_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Zomato Delivery Time Predictor",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&display=swap');

        html, body, [class*="css"] {{ font-family: 'Poppins', sans-serif; }}
        h1, h2, h3 {{ font-family: 'Poppins', sans-serif; font-weight: 700; }}

        div.stButton > button {{
            background-color: {BRAND_RED};
            color: white;
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            border-radius: 8px;
            border: none;
            padding: 0.6rem 1.2rem;
            width: 100%;
        }}
        div.stButton > button:hover {{ background-color: #a81a24; color: white; }}

        .result-card {{
            background: linear-gradient(135deg, {BRAND_RED}, #8f1420);
            padding: 1.5rem 2rem;
            border-radius: 14px;
            text-align: center;
            color: white;
            margin-top: 1rem;
        }}
        .result-card h2 {{ margin: 0; font-size: 2.4rem; color: white; }}
        .result-card p {{ margin: 0.2rem 0 0 0; opacity: 0.85; }}

        .app-header {{
            display: flex;
            align-items: center;
            gap: 1.2rem;
            margin-bottom: 0.5rem;
        }}
        .app-header img {{
            height: 72px;
            width: 72px;
            object-fit: contain;
            border-radius: 14px;
        }}
        .app-header h1 {{
            color: {BRAND_RED};
            margin: 0;
            font-weight: 800;
            line-height: 1.2;
        }}
        .app-header p {{
            opacity: 0.7;
            margin: 0.2rem 0 0 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()

# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts() -> dict:
    return {name: joblib.load(path) for name, path in ARTIFACT_PATHS.items()}


try:
    artifacts = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Missing artifact file: {e.filename}. "
        "Make sure all .pkl files from the training notebook are in this folder."
    )
    st.stop()

model = artifacts["model"]
scaler = artifacts["scaler"]
feature_columns: list[str] = artifacts["feature_columns"]
iqr_bounds: dict = artifacts["iqr_bounds"]
category_options: dict = artifacts["category_options"]
metrics: dict = artifacts.get("metrics", {})


# --------------------------------------------------------------------------
# Feature engineering helpers (mirrors the training notebook)
# --------------------------------------------------------------------------
def clip_to_training_bounds(value: float, column: str) -> float:
    """IQR capping — same rule as notebook Section 10."""
    lower, upper = iqr_bounds[column]
    return float(np.clip(value, lower, upper))


def build_feature_row(
    age: int, rating: float, traffic: str, festival: str, weather: str,
    order_type: str, vehicle_type: str, city: str,
    distance_km_input: float,
    order_date: date, order_time: time,
) -> pd.DataFrame:
    """Turn raw form inputs into the exact row shape the model expects."""
    distance_km = clip_to_training_bounds(distance_km_input, "distance_km")
    age = clip_to_training_bounds(age, "Delivery_person_Age")
    rating = clip_to_training_bounds(rating, "Delivery_person_Ratings")

    row = pd.Series(0.0, index=feature_columns, dtype=float)
    row["Delivery_person_Age"] = age
    row["Delivery_person_Ratings"] = rating
    row["distance_km"] = distance_km
    row["order_day_of_week"] = order_date.weekday()
    row["order_hour"] = order_time.hour
    row["Road_traffic_density"] = TRAFFIC_ORDER[traffic]
    row["Festival"] = 1 if festival == "Yes" else 0

    for column, value in [
        ("Weather_conditions", weather),
        ("Type_of_order", order_type),
        ("Type_of_vehicle", vehicle_type),
        ("City", city),
    ]:
        dummy_col = f"{column}_{value}"
        if dummy_col in row.index:  # absent => this was the drop_first baseline
            row[dummy_col] = 1

    return pd.DataFrame([row])[feature_columns]


def predict_delivery_time(features: pd.DataFrame) -> float:
    scaled = pd.DataFrame(scaler.transform(features), columns=feature_columns)
    return float(model.predict(scaled)[0])


# --------------------------------------------------------------------------
# Header (base64-embedded logo, flexbox-aligned with title)
# --------------------------------------------------------------------------
def render_header() -> None:
    if LOGO_PATH.exists():
        logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="logo">'
    else:
        logo_html = "<span style='font-size:3rem;'>🛵</span>"

    st.markdown(
        f"""
        <div class="app-header">
            {logo_html}
            <div>
                <h1>Zomato Delivery Time Predictor</h1>
                <p>Linear Regression model &middot; trained on order, weather, traffic and location features</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_header()
st.divider()

# --------------------------------------------------------------------------
# Sidebar — project info / model performance
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 About This Project")
    st.write(
        "An end-to-end regression pipeline: EDA, missing-value imputation, "
        "feature engineering (Haversine distance, time features), outlier "
        "treatment, VIF-checked encoding, and a scikit-learn Linear "
        "Regression model."
    )
    if metrics:
        st.subheader("Model performance (test set)")
        m1, m2 = st.columns(2)
        m1.metric("R²", f"{metrics.get('r2', 0):.3f}")
        m2.metric("Adj. R²", f"{metrics.get('adj_r2', 0):.3f}")
        m3, m4 = st.columns(2)
        m3.metric("RMSE", f"{metrics.get('rmse', 0):.1f} min")
        m4.metric("MAE", f"{metrics.get('mae', 0):.1f} min")
    st.caption("Built with scikit-learn & Streamlit")

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("🧾 Your Order")
    c1, c2, c3 = st.columns(3)
    order_type = c1.selectbox("Order Type", category_options["Type_of_order"])
    city = c2.selectbox("City Type", category_options["City"])
    distance_input = c3.number_input(
        "Distance to delivery location (km)",
        min_value=0.1, max_value=50.0, value=5.0, step=0.5,
    )

    with st.expander("⚙️ Delivery conditions (auto-filled by the system in production — adjust to test)"):
        st.caption(
            "In a live app these come from partner assignment, a weather API, "
            "a traffic API, and the current time — not from the customer. "
            "They're editable here so you can see how they move the prediction."
        )
        e1, e2, e3 = st.columns(3)
        weather = e1.selectbox("Weather", category_options["Weather_conditions"])
        traffic = e2.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"], index=1)
        festival = e3.selectbox("Festival Day?", ["No", "Yes"])

        e4, e5, e6 = st.columns(3)
        age = e4.number_input("Delivery Partner Age", min_value=15, max_value=65, value=30)
        rating = e5.slider("Delivery Partner Rating", 1.0, 5.0, 4.5, step=0.1)
        vehicle_type = e6.selectbox("Vehicle Type", category_options["Type_of_vehicle"])

        e7, e8 = st.columns(2)
        order_date = e7.date_input("Order Date", value=datetime.today())
        order_time = e8.time_input("Order Time", value=datetime.now().time())

    submitted = st.form_submit_button("🔮 Predict Delivery Time")

# --------------------------------------------------------------------------
# Prediction output
# --------------------------------------------------------------------------
if submitted:
    features = build_feature_row(
        age, rating, traffic, festival, weather, order_type, vehicle_type,
        city, distance_input, order_date, order_time,
    )
    prediction = predict_delivery_time(features)
    distance = features["distance_km"].iloc[0]

    st.markdown(
        f"""
        <div class="result-card">
            <p>ESTIMATED DELIVERY TIME</p>
            <h2>{prediction:.0f} minutes</h2>
            <p>Distance: {distance:.2f} km &middot; {order_time.strftime('%I:%M %p')} on {order_date.strftime('%A')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "<hr><p style='text-align:center; opacity:0.5; font-size:0.8rem;'>"
    "Portfolio project &middot; not affiliated with Zomato Media Pvt. Ltd."
    "</p>",
    unsafe_allow_html=True,
)
