import os
from model_pipeline import load_model

# Adjust path to go one level up
model_path = os.path.join(os.path.dirname(__file__), "../models/gbm_model.joblib")

print(f"Checking model path: {model_path}")

model = load_model(model_path)
print("✅ Model loaded successfully!")
