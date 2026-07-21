import pandas as pd
import joblib

# Load saved model and columns
model = joblib.load("heart_model.pkl")
columns = joblib.load("heart_columns.pkl")

# Sample patient data
sample_data = {
    'Age': 40,
    'Sex': 'M',
    'ChestPainType': 'ATA',
    'RestingBP': 120,
    'Cholesterol': 200,
    'FastingBS': 0,
    'RestingECG': 'Normal',
    'MaxHR': 150,
    'ExerciseAngina': 'N',
    'Oldpeak': 1.0,
    'ST_Slope': 'Up'
}

# Convert to DataFrame
sample_df = pd.DataFrame([sample_data])

# Apply One-Hot Encoding
sample_df = pd.get_dummies(sample_df)

# Match training columns
sample_df = sample_df.reindex(columns=columns, fill_value=0)

# Make prediction
prediction = model.predict(sample_df)

print("Prediction:", prediction[0])

if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")