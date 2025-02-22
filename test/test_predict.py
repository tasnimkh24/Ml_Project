from src.model_pipeline import load_model, predict
import numpy as np

def test_predict():
    model = load_model()
    sample_features = np.random.rand(1, 8)  # Données factices
    prediction = predict(sample_features)
    assert prediction in [0, 1]  # Vérifiez que la prédiction est valide
