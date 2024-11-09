import mlflow.sklearn
from pydantic import BaseModel


class ModelLoader:
    def __init__(self, model_uri: str):
        self.model = None
        self.model_uri = model_uri
        self.load_model()

    def load_model(self):
        self.model = mlflow.sklearn.load_model(self.model_uri)

    def predict(self, text: str):
        prediction = self.model.predict([text])
        return prediction[0]


class Complaint(BaseModel):
    text: str