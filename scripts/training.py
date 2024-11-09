import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
import yaml
from preprocess import nan_removal


# Load configuration from YAML file
with open('scripts\config.yaml') as config_file:
    config = yaml.safe_load(config_file)


def train_model():

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    version = config["mlflow"]["version"]
    with mlflow.start_run(run_name=f"Training Run {version}") as run:

        # Load and preprocess data
        data_raw = pd.read_csv(config["data"]["data_path"])
        data = nan_removal(data_raw)


        # Splitting data
        X = data['narrative']
        y = data['product']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        # Training the model with tokenizer and LR Pipeline using config parameters
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words=config["vectorizer"]["stop_words"],
                max_features=config["vectorizer"]["max_features"],
                ngram_range=tuple(config["vectorizer"]["ngram_range"])
            )),
            ('clf', LogisticRegression(
                max_iter=config["model"]["max_iter"], 
                penalty=config["model"]["penalty"], 
                random_state=config["model"]["random_state"],
                C=config["model"]["C"], 
                solver=config["model"]["solver"]
            ))
        ])
        pipeline.fit(X_train, y_train)


        # Log parameters one by one
        mlflow.log_param("model_max_iter", config["model"]["max_iter"])
        mlflow.log_param("model_penalty", config["model"]["penalty"])
        mlflow.log_param("model_random_state", config["model"]["random_state"])
        mlflow.log_param("model_C", config["model"]["C"])
        mlflow.log_param("model_solver", config["model"]["solver"])
        mlflow.log_param("vectorizer_stop_words", config["vectorizer"]["stop_words"])
        mlflow.log_param("vectorizer_max_features", config["vectorizer"]["max_features"])
        mlflow.log_param("vectorizer_ngram_range", config["vectorizer"]["ngram_range"])
        mlflow.log_param("mlflow_experiment_name", config["mlflow"]["experiment_name"])
        mlflow.log_param("mlflow_version", config["mlflow"]["version"])
        mlflow.log_param("data_path", config["data"]["data_path"])


        # Calculate and log metrics
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)


        # Log the model
        model_name = f"CustomerComplaintsModel_{version}"
        mlflow.sklearn.log_model(pipeline, artifact_path="model", registered_model_name=model_name)
        print(f"Model registered with run ID: {run.info.run_id}")


train_model()
#if __name__ == "__main__":
#    train_model()
