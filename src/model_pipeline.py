import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import mlflow
import mlflow.sklearn
import logging
import sklearn
from mlflow.tracking import MlflowClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(train_path, test_path):
    """
    Prepare the dataset for training and testing.
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, X_cluster_scaled, y_cluster
    """
    try:
        # 1. Load data
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

        # 2. Copy data for preparation
        df_prep = df.copy()

        # 3. Convert binary categorical variables
        df_prep['International plan'] = df_prep['International plan'].map({'Yes': 1, 'No': 0})
        df_prep['Voice mail plan'] = df_prep['Voice mail plan'].map({'Yes': 1, 'No': 0})
        df_prep['Churn'] = df_prep['Churn'].astype(int)

        # 4. Target Encoding for 'State'
        target_mean = df_prep.groupby('State')['Churn'].mean()
        df_prep['STATE_TargetMean'] = df_prep['State'].map(target_mean)

        # 5. Label Encoding for 'State'
        label_encoder = LabelEncoder()
        df_prep['STATE_Label'] = label_encoder.fit_transform(df_prep['State'])
        df_prep = df_prep.drop(columns=['State'])

        # 6. Remove highly correlated columns
        corr_data = df_prep.corr()
        upper_triangle = corr_data.where(np.triu(np.ones(corr_data.shape), k=1).astype(bool))
        high_correlation_columns = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]
        df_prep_dropped = df_prep.drop(columns=high_correlation_columns)

        # 7. Clip extreme values
        lower_limit = df_prep_dropped.quantile(0.05)
        upper_limit = df_prep_dropped.quantile(0.95)
        df_prep_clipped = df_prep_dropped.apply(lambda x: x.clip(lower_limit[x.name], upper_limit[x.name]))

        # 8. Separate features and target
        df_classif = df_prep_clipped.copy()
        X = df_classif.drop(columns=['Churn'])
        y = df_classif['Churn']

        df_cluster = df_prep_clipped.copy()
        X_cluster = df_cluster.drop(columns=['Churn'])
        y_cluster = df_cluster['Churn']

        # 9. Balance data with SMOTE
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        X_cluster, y_cluster = smote.fit_resample(X_cluster, y_cluster)

        # 10. Feature selection with ANOVA F-score
        F_scores, p_values = f_classif(X, y)
        scores_df = pd.DataFrame({'Feature': X.columns, 'F-Score': F_scores, 'P-Value': p_values})
        significant_features = scores_df[scores_df['P-Value'] < 0.05]['Feature'].tolist()

        # 11. Drop non-significant columns
        columns_to_drop = ['STATE_TargetMean', 'STATE_Label', 'Account length',
                           'Total night calls', 'Area code', 'Total day calls',
                           'Total eve calls']
        X = X.drop(columns=columns_to_drop, errors='ignore')
        X_cluster = X_cluster.drop(columns=columns_to_drop, errors='ignore')

        # 12. Split data into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 13. Normalize data with StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_cluster_scaled = scaler.fit_transform(X_cluster)

        # Save preprocessing objects
        preprocessing_objects = {
            'label_encoder': label_encoder,
            'scaler': scaler,
            'columns_to_drop': columns_to_drop,
            'target_mean': target_mean
        }
        joblib.dump(preprocessing_objects, "models/preprocessing_objects.joblib")

        # Save training columns
        joblib.dump(X.columns.tolist(), "models/training_columns.joblib")

        return X_train_scaled, X_test_scaled, y_train, y_test, X_cluster_scaled, y_cluster

    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        raise

def train_model(X_train, y_train):
    """
    Train a Gradient Boosting Classifier with hyperparameter optimization.
    """
    try:
        # Define hyperparameter search space
        param_dist = {
            'n_estimators': randint(50, 200),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }

        # Initialize the model
        gb_model = GradientBoostingClassifier(random_state=42)

        # Optimize hyperparameters with RandomizedSearchCV
        random_search = RandomizedSearchCV(
            gb_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=3,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1
        )

        # Train the model
        random_search.fit(X_train, y_train)

        # Log hyperparameters and metrics to MLflow
        best_params = random_search.best_params_
        mlflow.log_params(best_params)  # Log parameters
        mlflow.log_metric("accuracy", random_search.best_score_)  # Log metrics
        mlflow.sklearn.log_model(random_search.best_estimator_, "model")  # Log model
        logger.info("✅ Model trained and logged in MLflow!")

        # Save the model locally
        save_model(random_search.best_estimator_)

        # Return the best model
        return random_search.best_estimator_

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
    
    
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and log metrics.
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Plot and log confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # Print results
        logger.info(f"\n✅ Evaluation Completed!")
        logger.info(f"📊 Accuracy: {accuracy:.4f}")
        logger.info(f"📊 Classification Report:\n{report}")
        logger.info(f"📊 Confusion Matrix:\n{cm}")

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def save_model(model, filename="gbm_model.joblib"):
    """
    Save the model to the specified filename.
    """
    try:
        # Chemin relatif vers le dossier models/
        model_path = os.path.join("models", filename)
        # Créez le dossier models/ s'il n'existe pas
        os.makedirs("models", exist_ok=True)
        # Sauvegardez le modèle
        joblib.dump(model, model_path)
        logger.info(f"\n💾 Model saved to '{model_path}' and logged as an artifact.")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def load_model(filename="gbm_model.joblib"):
    """
    Load a trained model from the specified filename.
    """
    try:
        # If filename already includes the models/ directory, use it as is
        if filename.startswith("models/"):
            model_path = filename
        else:
            # Otherwise, construct the path relative to the models/ directory
            model_path = os.path.join("models", filename)
        
        # Check if the file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Log library versions
        logger.info(f"numpy version: {np.__version__}")
        logger.info(f"joblib version: {joblib.__version__}")
        logger.info(f"scikit-learn version: {sklearn.__version__}")

        # Load the model
        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        logger.info(f"\n📂 Model loaded from '{model_path}'")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def transition_model_stage(model_name, model_version, new_stage):
    """
    Transition a model version to a new stage.
    """
    try:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=new_stage,
        )
        logger.info(f"✅ Model version {model_version} transitioned to {new_stage} stage.")
    except Exception as e:
        logger.error(f"Error transitioning model stage: {e}")
        raise

def get_model_stage(model_name, model_version):
    """
    Get the current stage of a model version.
    """
    try:
        client = MlflowClient()
        model_version_details = client.get_model_version(name=model_name, version=model_version)
        return model_version_details.current_stage
    except Exception as e:
        logger.error(f"Error getting model stage: {e}")
        raise

def deploy_model(stage="Staging"):
    """
    Deploy the model, register it in MLflow, and transition it to a specific stage.
    """
    try:
        # Load the trained model
        model = load_model("models/gbm_model.joblib")
        logger.info("\n🚀 Deploying Model...")

        # Log the deployed model in MLflow
        with mlflow.start_run():
            logger.info("✅ MLflow run started.")
            
            # Log the model as an artifact
            mlflow.sklearn.log_model(model, "deployed_model")
            logger.info("✅ Model logged in MLflow.")

            # Register the model in the Models section
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/deployed_model"
            registered_model = mlflow.register_model(model_uri, "My_Deployed_Model")
            logger.info("✅ Model registered in MLflow Models section.")

            # Transition the model to the specified stage
            transition_model_stage(registered_model.name, registered_model.version, stage)
            logger.info(f"✅ Model transitioned to {stage} stage.")

        # Verify that the model was logged, registered, and transitioned
        logger.info("✅ Model deployment completed!")

    except Exception as e:
        logger.error(f"Error during model deployment: {e}")
        raise