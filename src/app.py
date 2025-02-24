from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the trained model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Remonte à projet_final
MODEL_PATH = os.path.join(BASE_DIR, "models", "gbm_model.joblib")
print(f"Trying to load model from: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print(f"⚠️ Model file not found at {MODEL_PATH}. Please retrain the model using the /retrain endpoint.")
    model = None  # Pour éviter une erreur lors du chargement

model = joblib.load(MODEL_PATH)

# Define the input data schema for prediction
class PredictionInput(BaseModel):
    features: list  # List of feature values for prediction

# Define the input data schema for retraining
class RetrainInput(BaseModel):
    data_path: str  # Path to the new dataset for retraining
    test_size: float = 0.2  # Test size for train-test split
    random_state: int = 42  # Random state for reproducibility

# Initialize FastAPI app
app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Prediction API!"}

# Prediction endpoint
@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convert input features to a numpy array
        features = np.array(input_data.features).reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(features)

        # Convert prediction to a readable message
        message = "The client may churn" if int(prediction[0]) == 1 else "The client may not churn"

        # Return the prediction message
        return {"prediction": message}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Retrain endpoint
@app.post("/retrain")
def retrain(input_data: RetrainInput):
    try:
        # Load the new dataset
        if not os.path.exists(input_data.data_path):
            raise FileNotFoundError(f"Dataset not found at {input_data.data_path}")
        
        df = pd.read_csv(input_data.data_path)
        
        # Preprocess the data (assuming the same preprocessing as during training)
        # Example: Convert binary categorical variables
        df['International plan'] = df['International plan'].map({'Yes': 1, 'No': 0})
        df['Voice mail plan'] = df['Voice mail plan'].map({'Yes': 1, 'No': 0})
        df['Churn'] = df['Churn'].astype(int)

        # Separate features and target
        X = df.drop(columns=['Churn'])
        y = df['Churn']

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=input_data.test_size, random_state=input_data.random_state
        )

        # Retrain the model
        new_model = GradientBoostingClassifier(random_state=input_data.random_state)
        new_model.fit(X_train, y_train)

        # Evaluate the new model
        y_pred = new_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save the new model
        joblib.dump(new_model, MODEL_PATH)

        return {
            "message": "Model retrained successfully!",
            "accuracy": accuracy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
