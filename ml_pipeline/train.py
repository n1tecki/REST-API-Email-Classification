# trainer.py
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from ml_pipeline.utils.preprocess import DataPreprocessor
from ml_pipeline.utils.validate import Validator
from ml_pipeline.utils.config_loader import load_config
from ml_pipeline.utils.mlflow_logger import MLFlowLogger



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
        with mlflow.start_run(run_name=f"Training Run for {self.config['model']['model_name']}") as run:
            self.load_data()
            X_train, X_test, y_train, y_test = self.split_data()
            self.build_pipeline()
            self.logger.log_parameters(self.pipeline)
            self.train(X_train, y_train)
            validator = Validator(self.pipeline)
            metrics = validator.evaluate(X_test, y_test)
            self.logger.log_metrics(metrics)
            self.logger.log_model_and_transition(self.pipeline, "model", self.config["model"]["model_name"])
            print(f"Model version {run.info.run_id} logged'.")



if __name__ == "__main__":
    trainer = Trainer('ml_pipeline\model_config.yaml')
    trainer.run()