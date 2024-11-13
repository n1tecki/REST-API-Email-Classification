import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient



class MLFlowLogger:
    def __init__(self, config):
        self.config = config
        self.client = MlflowClient()



    def log_parameters(self, pipeline):
        params = {
            "model_max_iter": self.config["model"]["model_param"]["max_iter"],
            "model_penalty": self.config["model"]["model_param"]["penalty"],
            "model_random_state": self.config["model"]["model_param"]["random_state"],
            "model_C": self.config["model"]["model_param"]["C"],
            "model_solver": self.config["model"]["model_param"]["solver"],
            "vectorizer_stop_words": self.config["vectorizer"]["stop_words"],
            "vectorizer_max_features": self.config["vectorizer"]["max_features"],
            "vectorizer_ngram_range": self.config["vectorizer"]["ngram_range"],
            "mlflow_experiment_name": self.config["mlflow"]["experiment_name"],
            "data_path": self.config["data"]["data_path"],
            "model_type": type(pipeline.named_steps['clf']).__name__,
            "vectorizer_type": type(pipeline.named_steps['tfidf']).__name__ 
        }
        for param, value in params.items():
            mlflow.log_param(param, value)



    def log_metrics(self, metrics):
        mlflow.log_metrics(metrics)



    def log_model_and_transition(self, model, artifact_path, model_name):
        mlflow.sklearn.log_model(model, artifact_path=artifact_path, registered_model_name=model_name)

