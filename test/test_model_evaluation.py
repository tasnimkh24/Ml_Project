from src.model_pipeline import evaluate_model
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

def test_evaluate_model():
    # Créer des données factices pour l'entraînement et le test
    X_train = np.random.rand(100, 8)  # 100 échantillons, 8 caractéristiques
    y_train = np.random.randint(0, 2, 100)  # Labels binaires (0 ou 1)
    X_test = np.random.rand(50, 8)  # 50 échantillons, 8 caractéristiques
    y_test = np.random.randint(0, 2, 50)  # Labels binaires (0 ou 1)

    # Entraîner le modèle
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # Évaluer le modèle
    evaluate_model(model, X_test, y_test)  # Vérifiez qu'il n'y a pas d'erreur
