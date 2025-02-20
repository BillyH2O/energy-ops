import os
import mlflow
from mlflow.tracking import MlflowClient


def push_to_model_registry(registry_name: str, run_id: str):
    """
    Envoie un modèle enregistré dans le tracking MLflow vers le registre de modèles.
    Retourne la version du modèle enregistrée.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_SERVER"))
    result = mlflow.register_model(
        f"runs:/{run_id}/model", registry_name  # Notez le chemin "model" ici
    )
    return result.version


def stage_model(registry_name: str, version: int):
    """
    Transitionne une version du modèle vers l'état staging ou production selon ENV.
    """
    env = os.getenv("ENV")
    if env not in ["staging", "production"]:
        return

    client = MlflowClient()
    client.transition_model_version_stage(
        name=registry_name,
        version=int(version),
        stage=env[0].upper() + env[1:],  # Capitalise la première lettre (ex: "Staging")
    )
