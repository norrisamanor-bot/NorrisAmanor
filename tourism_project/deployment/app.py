import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_predictor_model_v1.joblib"

st.sidebar.write(f"Model path: {MODEL_PATH}")
st.sidebar.write(f"Model exists: {MODEL_PATH.exists()}")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Streamlit App
# --------------------------------------------------
st.title("Tourism Package Prediction")
st.write(
    "Fill in the customer details below to predict whether the customer "
    "is likely to purchase a travel package."
)

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
Age = st.slider("Age", 18, 70, 30)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

DurationOfPitch = st.slider(
    "Duration of Pitch (Minutes)",
    0,
    100,
    15
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Others"]
)

NumberOfPersonVisiting = st.slider(
    "Number of Persons Visiting",
    1,
    5,
    2
)

NumberOfFollowups = st.slider(
    "Number of Follow-ups",
    1,
    10,
    3
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [1, 2, 3, 4, 5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Divorced", "Unmarried"]
)

NumberOfTrips = st.slider(
    "Number of Trips",
    1,
    20,
    3
)

Passport = st.selectbox(
    "Has Passport?",
    ["Yes", "No"]
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

OwnCar = st.selectbox(
    "Owns a Car?",
    ["Yes", "No"]
)

NumberOfChildrenVisiting = st.slider(
    "Number of Children Visiting",
    0,
    5,
    1
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "AVP", "VP", "Sr. Manager"]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=1000.0,
    value=30000.0
)

# --------------------------------------------------
# Build Input DataFrame
# --------------------------------------------------
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
CLASSIFICATION_THRESHOLD = 0.55

if st.button("Predict"):

    probability = model.predict_proba(input_data)[0, 1]

    prediction = int(
        probability >= CLASSIFICATION_THRESHOLD
    )

    if prediction == 1:
        st.success(
            f"Customer is likely to purchase the travel package "
            f"(Probability: {probability:.2%})"
        )
    else:
        st.warning(
            f"Customer is unlikely to purchase the travel package "
            f"(Probability: {probability:.2%})"
        )
