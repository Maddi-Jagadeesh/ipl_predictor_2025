import streamlit as st
import pickle
import pandas as pd
import os

# =========================
# Load Saved Model & Encoder
# =========================
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏")

# Load model
MODEL_PATH = 'model.pkl'
ENCODER_PATH = 'team_encoder.pkl'

if not os.path.exists(MODEL_PATH):
    st.error("❌ model.pkl not found. Please place the trained model file in this folder.")
    st.stop()

if not os.path.exists(ENCODER_PATH):
    st.error("❌ team_encoder.pkl not found. Please place the saved team_encoder file in this folder.")
    st.stop()

model = pickle.load(open(MODEL_PATH, 'rb'))
team_encoder = pickle.load(open(ENCODER_PATH, 'rb'))

# =========================
# Streamlit App UI
# =========================
st.title("🏏 IPL Win Probability Predictor")
st.markdown("### Enter the current match details to get the win probability.")

# Select Teams
teams = list(team_encoder.classes_)
batting_team = st.selectbox("Select Batting Team", teams)
bowling_team = st.selectbox("Select Bowling Team", teams)

# Prevent same team selection
if batting_team == bowling_team:
    st.warning("⚠️ Batting and bowling teams should not be the same.")

# Numeric Inputs
runs_left = st.number_input("Runs Left", min_value=0, step=1)
balls_left = st.number_input("Balls Left", min_value=0, step=1)
wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10, step=1)
crr = st.number_input("Current Run Rate (CRR)", min_value=0.0, step=0.1, format="%.2f")
rrr = st.number_input("Required Run Rate (RRR)", min_value=0.0, step=0.1, format="%.2f")

# =========================
# Prediction
# =========================
if st.button("Predict Win Probability"):
    try:
        # Encode Teams
        batting_team_enc = team_encoder.transform([batting_team])[0]
        bowling_team_enc = team_encoder.transform([bowling_team])[0]

        # Prepare Input DataFrame
        input_df = pd.DataFrame({
            'balls_left': [balls_left],
            'runs_left': [runs_left],
            'wickets_left': [wickets_left],
            'crr': [crr],
            'rrr': [rrr],
            'batting_team_enc': [batting_team_enc],
            'bowling_team_enc': [bowling_team_enc]
        })

        # Check if model supports predict_proba
        if hasattr(model, "predict_proba"):
            win_prob = model.predict_proba(input_df)[0][1] * 100
            loss_prob = 100 - win_prob
            st.success(f"📊 Win Probability for {batting_team}: {win_prob:.2f}%")
            st.info(f"📉 Loss Probability: {loss_prob:.2f}%")
        else:
            # Fallback to prediction without probability
            prediction = model.predict(input_df)[0]
            if prediction == 1:
                st.success(f"✅ Predicted Winner: {batting_team}")
            else:
                st.error(f"❌ Predicted Winner: {bowling_team}")

    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
