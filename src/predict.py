import json
from pathlib import Path
import joblib
import pandas as pd

# Welcome, in this function we are going to
# validate and do our predictions

# ----------------------------------
# PATHS
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "logistic_regression_cv.pkl"
FEATURES_PATH =BASE_DIR / "data" / "processed" / "feature_info.json"

# ----------------------------------
# Loading model and feature information
# ----------------------------------

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r", encoding="utf-8") as file:
    feature_info = json.load(file)

FEATURES = list(feature_info.keys())

def validate_structure(data):
    """
    In this function we are going to check that the input
    contains exactly what we need to predict
    """
    
    input_features = set(data.keys())
    expected_features = set(FEATURES)
    
    # Missing features
    missing  = expected_features - input_features
    
    if missing:
        raise ValueError(
            f"Missing features: {sorted(missing)}"
        )
    
    # Unexpected features
    extra = input_features - expected_features
    if extra:
            raise ValueError(
                f"Unexpected features: {sorted(extra)}"
            )
    
    
def validate_categorical_features(data):
    """
    Check that categorical features contain valid
    values according to the JSON dictionary
    """
    for feature, info in feature_info.items():
        
        if info["type"] != "categorical":
            continue
        
        allowed_values = {
            int(value)
            for value in info["values"].keys()
        }
        
        value = data[feature]
        
        # Check if the value is numerical
        if isinstance(value, bool):
            raise ValueError(
                f"{feature} must be an integer category"
            )
        
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{feature} must be an integer category"
            )
        
        if value not in allowed_values:
            raise ValueError(
                f"Invalid value for '{feature}': {value}."
                f"Allowed values: {sorted(allowed_values)}"
            )
        
        data[feature] = value

def validate_numeric_features(data):
    """
    Validate numerical features and also apply
    known constraints from the dataset documentation
    """
    
    for feature, info in feature_info.items():
        
        if info["type"] != "numerical": continue
        
        value = data[feature]
        
        # Check for any missing value
        if value is None:
            raise ValueError(
                f"{feature} cannot be empty"
            )
        
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{feature} must be a numerical number"
            )
        
        # Check for NaN and infinity
        if pd.isna(value) or value in (float("inf"), float("-inf")):
            raise ValueError(
                f"{feature} must be a finite numerical value."
                )
        
        # Age validation
        if feature == "Age at enrollment":
            if value <= 0 or value >= 100:
                raise ValueError(
                    "Age at enrollment must be between 1 and 99."
                )
        
        # Grade specific validations
        if "grade" in feature.lower():
            if value < 0 or value > 20:
                raise ValueError(
                    f"{feature} must be between 0 and 20"
                )
        
        if ( "credited" in feature.lower()
            or "enrolled" in feature.lower()
            or "evaluations" in feature.lower()
            or "approved" in feature.lower()
            or "without evaluations" in feature.lower()
        ):
            if value < 0 or value != int(value):
                raise ValueError(
                    f"{feature} must be a non-negative integer."
                )
        
        data[feature] = value

def validate_data(data):
    """
    Here we are going to run all
    the validation checks
    """
    
    if not isinstance(data, dict):
        raise TypeError(
            "Input data must be provided as a dictionary"
        )
    
    # Structure
    validate_structure(data)
    
    # Create a copy to avoid
    # modifying the original dictionary
    
    data = data.copy()
    
    # Values
    validate_categorical_features(data)
    validate_numeric_features(data)
    
    # And finally we force the exact order expected
    # by the model
    data = {
        feature: data[feature]
        for feature in FEATURES
    }
    
    return data


# -------------------------------
# Predictions
# -------------------------------

def predict_student(data):
    """
    Here we are going to validate all the student data and return
    our prediction
    """
    
    data = validate_data(data)
    
    df = pd.DataFrame([data])
    
    prediction = int(model.predict(df)[0])
    
    probabilities = model.predict_proba(df)[0]
    
    probability_dropout = float(probabilities[1])
    probability_non_dropout = float(probabilities[0])

    if prediction == 1:
        label = "Dropout"
    else:
        label = "Graduate/Enrolled"
    
    return {
        "prediction": prediction,
        "label": label,
        "probability_dropout" : probability_dropout,
        "probability_non_dropout" : probability_non_dropout
    
    }
    
# EXAMPLE

if __name__ == "__main__":

    student ={
    "Marital status": 1,
    "Application mode": 8,
    "Application order": 5,
    "Course": 2,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Nacionality": 1,
    "Mother's qualification": 13,
    "Father's qualification": 10,
    "Mother's occupation": 6,
    "Father's occupation": 10,
    "Displaced": 1,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 1,
    "Scholarship holder": 0,
    "Age at enrollment": 20,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 0,
    "Curricular units 1st sem (evaluations)": 0,
    "Curricular units 1st sem (approved)": 0,
    "Curricular units 1st sem (grade)": 0.0,
    "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0,
    "Curricular units 2nd sem (enrolled)": 0,
    "Curricular units 2nd sem (evaluations)": 0,
    "Curricular units 2nd sem (approved)": 0,
    "Curricular units 2nd sem (grade)": 0.0,
    "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 10.8,
    "Inflation rate": 1.4,
    "GDP": 1.74,
    }

    # Expected prediction: 1
    result = predict_student(student)
    print(result)