
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Transfer Fee Predictor",
    page_icon="⚽",
    layout="wide"
)


# ---------------------------------
# Custom Styling
# ---------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------
# Project Paths
# ---------------------------------

project_folder = Path(__file__).resolve().parent.parent

model_path = project_folder / "models" / "transfer_fee_model.pkl"
data_path = project_folder / "data" / "cleaned_transfers.csv"


# ---------------------------------
# Load Model and Dataset
# ---------------------------------

try:
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

except FileNotFoundError:
    st.error("Model or dataset file not found.")
    st.stop()


# ---------------------------------
# Session State for History
# ---------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ---------------------------------
# Application Header
# ---------------------------------

st.markdown(
    '<div class="main-title">⚽ Premier League Transfer Fee Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Football Transfer Prediction</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------------------------
# Sidebar
# ---------------------------------

with st.sidebar:

    st.header("📌 About Project")

    st.write(
        "This application uses Machine Learning "
        "to estimate football player transfer fees."
    )

    st.info(
        "The result is an estimated value, "
        "not an official transfer fee."
    )

    st.header("📊 Dataset Information")

    st.write(f"Total Records: {len(df)}")


# ---------------------------------
# Player Information
# ---------------------------------

st.header("👤 Player Information")

player_name = st.text_input(
    "Player Name",
    placeholder="Enter player name"
)


# ---------------------------------
# Input Columns
# ---------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Player Age",
        min_value=16,
        max_value=45,
        value=24
    )

    position_options = sorted(
        df["position"].dropna().unique()
    )

    position = st.selectbox(
        "Player Position",
        position_options
    )


with col2:

    club_options = sorted(
        df["club_name"].dropna().unique()
    )

    club_name = st.selectbox(
        "Buying Club",
        club_options
    )

    source_club_options = sorted(
        df["club_involved_name"].dropna().unique()
    )

    club_involved_name = st.selectbox(
        "Previous Club",
        source_club_options
    )


with col3:

    transfer_period_options = sorted(
        df["transfer_period"].dropna().unique()
    )

    transfer_period = st.selectbox(
        "Transfer Period",
        transfer_period_options
    )

    year = st.number_input(
        "Transfer Year",
        min_value=1992,
        max_value=2026,
        value=2022
    )

    country_options = sorted(
        df["country"].dropna().unique()
    )

    country = st.selectbox(
        "Player Country",
        country_options
    )


# ---------------------------------
# Prediction
# ---------------------------------

st.divider()

predict_button = st.button(
    "🔮 Predict Transfer Fee",
    use_container_width=True
)


if predict_button:

    if player_name.strip() == "":
        st.warning("Please enter the player name.")

    else:

        input_data = pd.DataFrame({

            "age": [age],

            "position": [position],

            "club_name": [club_name],

            "club_involved_name": [club_involved_name],

            "transfer_period": [transfer_period],

            "year": [year],

            "country": [country]

        })

        try:

            prediction = model.predict(input_data)

            predicted_fee = max(0, prediction[0])

            # Save prediction history
            history_item = {

                "Player Name": player_name,

                "Age": age,

                "Position": position,

                "Buying Club": club_name,

                "Previous Club": club_involved_name,

                "Transfer Fee (€ Million)": round(
                    predicted_fee, 2
                )

            }

            st.session_state.prediction_history.append(
                history_item
            )

            # Display prediction
            st.success(
                f"💰 Estimated Transfer Fee: "
                f"€{predicted_fee:.2f} million"
            )

            st.info(
                f"Player: {player_name}"
            )

            st.warning(
                "This is an ML-based estimate, "
                "not an official transfer fee."
            )

        except Exception as error:

            st.error(
                "An error occurred during prediction."
            )

            st.code(str(error))


# ---------------------------------
# Prediction History
# ---------------------------------

st.divider()

st.header("📊 Prediction History")

if len(st.session_state.prediction_history) > 0:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("🗑️ Clear Prediction History"):

        st.session_state.prediction_history = []

        st.rerun()

else:

    st.info(
        "No predictions yet. "
        "Make a prediction to see your history here."
    )


# ---------------------------------
# Footer
# ---------------------------------

st.divider()

st.caption(
    "Premier League Transfer Fee Predictor | "
    "Machine Learning Project"
)