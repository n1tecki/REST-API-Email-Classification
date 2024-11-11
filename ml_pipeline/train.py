import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import yaml
from preprocess import DataPreprocessor
from validate import Validator



class Trainer:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.data = None
        self.pipeline = None
        self.preprocessor = DataPreprocessor()


    def load_config(self, path):
        with open(path) as file:
            return yaml.safe_load(file)


    def load_data(self):
        data_raw = pd.read_csv(self.config["data"]["data_path"])
        self.data = self.preprocessor.preprocess(data_raw)
        return self.data


    def split_data(self):
        X = self.data['narrative']
        y = self.data['product']
        return train_test_split(X, y, test_size=0.2, random_state=42)


    def build_pipeline(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words=self.config["vectorizer"]["stop_words"],
                max_features=self.config["vectorizer"]["max_features"],
                ngram_range=tuple(self.config["vectorizer"]["ngram_range"])
            )),
            ('clf', LogisticRegression(
                max_iter=self.config["model"]["max_iter"], 
                penalty=self.config["model"]["penalty"], 
                random_state=self.config["model"]["random_state"],
                C=self.config["model"]["C"], 
                solver=self.config["model"]["solver"]
            ))
        ])


    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)


    def log_metrics(self, metrics):
        mlflow.log_metrics(metrics)


    def log_parameters(self):
        mlflow.log_param("model_max_iter", self.config["model"]["max_iter"])
        mlflow.log_param("model_penalty", self.config["model"]["penalty"])
        mlflow.log_param("model_random_state", self.config["model"]["random_state"])
        mlflow.log_param("model_C", self.config["model"]["C"])
        mlflow.log_param("model_solver", self.config["model"]["solver"])
        mlflow.log_param("vectorizer_stop_words", self.config["vectorizer"]["stop_words"])
        mlflow.log_param("vectorizer_max_features", self.config["vectorizer"]["max_features"])
        mlflow.log_param("vectorizer_ngram_range", self.config["vectorizer"]["ngram_range"])
        mlflow.log_param("mlflow_experiment_name", self.config["mlflow"]["experiment_name"])
        mlflow.log_param("data_path", self.config["data"]["data_path"])


    def run(self):
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name=f"Training Run") as run:
            self.load_data()
            X_train, X_test, y_train, y_test = self.split_data()
            self.build_pipeline()
            self.train(X_train, y_train)

            validator = Validator(self.pipeline)
            metrics = validator.evaluate(X_test, y_test)
            self.log_metrics(metrics)
            self.log_parameters()

            model_name = f"CustomerComplaintsModel"
            mlflow.sklearn.log_model(self.pipeline, artifact_path="model", registered_model_name=model_name)

            client = MlflowClient()
            mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
            latest_version = client.get_registered_model(model_name).latest_versions[0].version
            client.transition_model_version_stage(
                name=model_name, version=latest_version, stage=self.config["mlflow"]["stage"]
            )

            print(f"Model registered with run ID: {run.info.run_id}")


trainer = Trainer('ml_pipeline/config.yaml')
trainer.run()
