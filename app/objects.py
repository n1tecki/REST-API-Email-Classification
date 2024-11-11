from mlflow.pyfunc import load_model
from mlflow.tracking import MlflowClient
from pydantic import BaseModel


class ModelLoader:
    def __init__(self, model_uri: str):
        self.model_uri = model_uri
        self.model = self._load_model()
        self.model_version = self._get_model_version()

    def _load_model(self):
        return load_model(model_uri=self.model_uri)

    def _get_model_version(self):
        client = MlflowClient()
        model_name = self.model_uri.split("/")[1]
        versions = client.get_latest_versions(name=model_name, stages=["Production"])
        return versions[0].version if versions else "unknown"

    def predict(self, text):
        return self.model.predict([text])[0]


class Complaint(BaseModel):
    text: str