# trainer.py
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import DataPreprocessor
from validate import Validator
from utils.config_loader import load_config
from utils.mlflow_logger import MLFlowLogger



class Trainer:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.preprocessor = DataPreprocessor()
        self.logger = MLFlowLogger(self.config)
        self.pipeline = None
        self.data = None



    def load_data(self):
        data_raw = pd.read_csv(self.config["data"]["data_path"])
        self.data = self.preprocessor.preprocess(data_raw)



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
                max_iter=self.config["model"]["model_param"]["max_iter"], 
                penalty=self.config["model"]["model_param"]["penalty"], 
                random_state=self.config["model"]["model_param"]["random_state"],
                C=self.config["model"]["model_param"]["C"], 
                solver=self.config["model"]["model_param"]["solver"]
            ))
        ])



    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)



    def run(self):
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name="Training Run") as run:
            self.load_data()
            X_train, X_test, y_train, y_test = self.split_data()
            self.build_pipeline()
            self.train(X_train, y_train)

            validator = Validator(self.pipeline)
            metrics = validator.evaluate(X_test, y_test)
            
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
                "data_path": self.config["data"]["data_path"]
            }
            self.logger.log_parameters(params)
            self.logger.log_metrics(metrics)
            self.logger.log_model_and_transition(self.pipeline, "model", self.config["model"]["model_name"], "Staging")



if __name__ == "__main__":
    trainer = Trainer('model_config.yaml')
    trainer.run()