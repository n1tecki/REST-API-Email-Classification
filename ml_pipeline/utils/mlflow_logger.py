import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient



class MLFlowLogger:
    def __init__(self, config):
        self.config = config
        self.client = MlflowClient()



    def log_parameters(self, params):
        for param, value in params.items():
            mlflow.log_param(param, value)



    def log_metrics(self, metrics):
        mlflow.log_metrics(metrics)



    def log_model_and_transition(self, model, artifact_path, model_name, stage):
        mlflow.sklearn.log_model(model, artifact_path=artifact_path, registered_model_name=model_name)
        latest_versions = self.client.get_latest_versions(name=model_name, stages=["None"])
        if latest_versions:
            latest_version = latest_versions[0].version
            self.client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage=stage
            )
            
            print(f"Model version {latest_version} transitioned to 'Staging'.")
