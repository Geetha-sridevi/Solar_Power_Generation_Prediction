
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# -----------------------------------------
# 1. Page Configuration
# -----------------------------------------

st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="☀️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "solar_power_model.pkl"
DATA_PATH = BASE_DIR / "Renewable.csv"

TARGET = "Energy delta[Wh]"

st.title("☀️ Solar Power Generation Prediction")

st.write(
    "Predict solar energy generation using "
    "weather conditions and time-related features."
)

# -----------------------------------------
# 2. Check Required Files
# -----------------------------------------

if not MODEL_PATH.exists():
    st.error("solar_power_model.pkl not found!")
    st.stop()

if not DATA_PATH.exists():
    st.error("Renewable.csv not found!")
    st.stop()


# -----------------------------------------
# 3. Load Model and Dataset
# -----------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():

    df = pd.read_csv(DATA_PATH)

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    df = df.dropna(subset=[TARGET])
    df = df.drop_duplicates()

    # Create cyclical features
    df['hour_sin'] = np.sin(
        2 * np.pi * df['hour'] / 24
    )

    df['hour_cos'] = np.cos(
        2 * np.pi * df['hour'] / 24
    )

    df['month_sin'] = np.sin(
        2 * np.pi * df['month'] / 12
    )

    df['month_cos'] = np.cos(
        2 * np.pi * df['month'] / 12
    )

    return df


try:
    model = load_model()
    df = load_dataset()

except Exception as e:
    st.error(f"Loading error: {e}")
    st.stop()


# -----------------------------------------
# 4. Prepare Input Features
# -----------------------------------------

X = df.drop(
    columns=['Energy delta[Wh]', 'Time']
)

feature_columns = X.columns.tolist()

categorical_columns = ['weather_type']

numerical_columns = [
    col for col in feature_columns
    if col not in categorical_columns
]


# -----------------------------------------
# 5. Date and Time Input
# -----------------------------------------

st.subheader("Select Date and Time")

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Select Date",
        value=datetime.now().date()
    )

with col2:
    now = datetime.now()

    selected_time = st.time_input(
        "Select Time",
        value=now.replace(
            minute=(now.minute // 15) * 15,
            second=0,
            microsecond=0
        ).time()
    )

selected_datetime = datetime.combine(
    selected_date,
    selected_time
)

hour = selected_datetime.hour
month = selected_datetime.month


# -----------------------------------------
# 6. User Input
# -----------------------------------------

st.subheader("Enter Weather Conditions")

input_values = {}

col1, col2 = st.columns(2)

for index, column in enumerate(numerical_columns):

    # Hour and month are derived from date/time
    if column == 'hour':
        input_values[column] = hour
        continue

    if column == 'month':
        input_values[column] = month
        continue

    # Cyclical features are calculated separately
    if column in [
        'hour_sin',
        'hour_cos',
        'month_sin',
        'month_cos'
    ]:
        continue

    series = pd.to_numeric(
        df[column],
        errors='coerce'
    ).dropna()

    if series.empty:
        minimum = 0.0
        maximum = 1.0
        default = 0.0

    else:
        minimum = float(series.min())
        maximum = float(series.max())
        default = float(series.median())

    if minimum == maximum:
        minimum -= 1
        maximum += 1

    selected_column = col1 if index % 2 == 0 else col2

    with selected_column:

        input_values[column] = st.number_input(
            column,
            min_value=minimum,
            max_value=maximum,
            value=min(max(default, minimum), maximum),
            format="%.4f"
        )


# -----------------------------------------
# 7. Weather Type Selection
# -----------------------------------------

weather_options = sorted(
    df['weather_type']
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

if not weather_options:
    weather_options = ["Unknown"]

input_values['weather_type'] = st.selectbox(
    "Weather Type",
    weather_options
)


# -----------------------------------------
# 8. Create Cyclical Features
# -----------------------------------------

input_values['hour_sin'] = np.sin(
    2 * np.pi * hour / 24
)

input_values['hour_cos'] = np.cos(
    2 * np.pi * hour / 24
)

input_values['month_sin'] = np.sin(
    2 * np.pi * month / 12
)

input_values['month_cos'] = np.cos(
    2 * np.pi * month / 12
)


# -----------------------------------------
# 9. Create Input DataFrame
# -----------------------------------------

input_df = pd.DataFrame(
    [[input_values[col] for col in feature_columns]],
    columns=feature_columns
)


# -----------------------------------------
# 10. Prediction
# -----------------------------------------

st.subheader("Predict Solar Energy Generation")

if st.button("Predict Energy Generation"):

    try:

        prediction = model.predict(input_df)[0]

        prediction = max(float(prediction), 0)

        prediction_kwh = prediction / 1000

        st.success("Prediction Completed!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Energy (Wh)",
                f"{prediction:,.2f} Wh"
            )

        with col2:
            st.metric(
                "Predicted Energy (kWh)",
                f"{prediction_kwh:,.4f} kWh"
            )

        st.write(
            "Selected Date and Time:",
            selected_datetime.strftime("%d-%m-%Y %H:%M")
        )

        with st.expander("View Input Details"):
            st.dataframe(
                input_df,
                use_container_width=True
            )

    except Exception as e:

        st.error(f"Prediction Error: {e}")


# -----------------------------------------
# 11. Footer
# -----------------------------------------

st.divider()

st.caption(
    "Solar Power Generation Prediction "
    "| Machine Learning Regression Project"
)
