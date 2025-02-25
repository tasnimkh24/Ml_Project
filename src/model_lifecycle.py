# model_lifecycle.py
import mlflow
from mlflow.tracking import MlflowClient

def transition_model_stage(model_name, model_version, new_stage):
    """
    Transition a model version to a new stage.
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=new_stage,
    )
    print(f"✅ Model version {model_version} transitioned to {new_stage} stage.")

def get_model_stage(model_name, model_version):
    """
    Get the current stage of a model version.
    """
    client = MlflowClient()
    model_version_details = client.get_model_version(name=model_name, version=model_version)
    return model_version_details.current_stage

if __name__ == "__main__":
    # Example usage
    model_name = "My_Deployed_Model"
    model_version = 1
    new_stage = "Staging"

    # Transition the model to a new stage
    transition_model_stage(model_name, model_version, new_stage)

    # Verify the stage
    current_stage = get_model_stage(model_name, model_version)
    print(f"Current stage of model version {model_version}: {current_stage}")
