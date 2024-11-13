from mlflow.pyfunc import load_model
from mlflow.tracking import MlflowClient
from pydantic import BaseModel



class ModelLoader:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_uri = self._get_latest_model_uri()
        self.model = self._load_model()
        self.model_version = self._get_model_version()



    def _get_latest_model_uri(self):
        client = MlflowClient()
        versions = client.get_latest_versions(name=self.model_name, stages=["Production"])
        if versions:
            latest_version = versions[0].version
            return f"models:/{self.model_name}/{latest_version}"
        else:
            raise ValueError("No versions found for the specified model in Production stage.")



    def _load_model(self):
        return load_model(model_uri=self.model_uri)



    def _get_model_version(self):
        return self.model_uri.split("/")[-1]



    def predict(self, text):
        return self.model.predict([text])[0]



class Complaint(BaseModel):
    text: str
