# src/app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import streamlit.components.v1 as components
import json

# Resolve project root and paths
ROOT = Path(__file__).resolve().parents[1]  # project root
MODEL_PATH = ROOT / "model" / "model_pipeline.pkl"
ASSETS_DIR = ROOT / "assets"

st.set_page_config(page_title="Netflix Churn Predictor", layout="centered")
st.title("🎬 Netflix Customer Churn Predictor (Advanced)")
st.markdown("Enter customer details to get churn prediction. The model and insights were trained in a Jupyter notebook.")

# Load model safely and cache it
@st.cache_resource
def load_model(path):
    if not path.exists():
        return None
    return joblib.load(path)

model = load_model(MODEL_PATH)
if model is None:
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

# --- Input form ---
with st.form("input_form"):
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    subscription_type = st.selectbox("Subscription Type", options=["Basic", "Standard", "Premium"])
    watch_hours = st.number_input("Watch Hours (per week)", min_value=0.0, value=10.0)
    avg_watch_time_per_day = st.number_input("Avg watch time per day (hours)", min_value=0.0, value=0.5)
    last_login_days = st.number_input("Days since last login", min_value=0, value=7)
    region = st.selectbox("Region", options=["Asia", "Europe", "North America", "South America", "Africa", "Oceania"])
    device = st.selectbox("Device", options=["TV", "Mobile", "Laptop", "Tablet"])
    monthly_fee = st.number_input("Monthly Fee", min_value=0.0, value=13.99)
    payment_method = st.selectbox("Payment Method", options=["Debit Card", "Credit Card", "PayPal", "Gift Card", "Crypto"])
    number_of_profiles = st.number_input("Number of Profiles", min_value=1, max_value=10, value=1)
    favorite_genre = st.selectbox("Favorite Genre", options=["Action","Drama","Comedy","Romance","Horror","Sci-Fi","Documentary","Other"])
    submitted = st.form_submit_button("Predict")

# --- Prediction logic (creates engineered feature used in training) ---
if submitted:
    # build input dataframe from form values
    input_df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "subscription_type": subscription_type,
        "watch_hours": watch_hours,
        "avg_watch_time_per_day": avg_watch_time_per_day,
        "last_login_days": last_login_days,
        "region": region,
        "device": device,
        "monthly_fee": monthly_fee,
        "payment_method": payment_method,
        "number_of_profiles": number_of_profiles,
        "favorite_genre": favorite_genre
    }])

    # Load metadata for engineered features (median used when training)
    metadata_path = ASSETS_DIR / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            median_val = float(meta.get("avg_watch_time_per_day_median", meta.get("avg_watch_time_median", 0.5)))
        except Exception:
            median_val = 0.5
    else:
        # fallback default if metadata missing
        median_val = 0.5

    # Create the engineered column exactly like in training
    input_df['low_engagement'] = (input_df['avg_watch_time_per_day'] < median_val).astype(int)

    # Predict churn using the saved pipeline (includes preprocessing)
    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
    else:
        if pred == 1:
            st.error(f"⚠️ Likely to churn — Risk: {prob:.2f}")
        else:
            st.success(f"✅ Likely to stay — Retention prob: {1-prob:.2f}")

st.markdown("---")
st.subheader("Model Insights")

# Embed saved Plotly assets if present
fi_html = ASSETS_DIR / "feature_importance.html"
roc_html = ASSETS_DIR / "roc_curve.html"

if fi_html.exists():
    st.markdown("**Feature Importance**")
    try:
        components.html(fi_html.read_text(encoding='utf-8'), height=420)
    except Exception as e:
        st.write("Could not embed feature importance HTML:", e)

if roc_html.exists():
    st.markdown("**ROC Curve (on test set)**")
    try:
        components.html(roc_html.read_text(encoding='utf-8'), height=420)
    except Exception as e:
        st.write("Could not embed ROC HTML:", e)

st.markdown("Assets are loaded from the `assets/` folder.")