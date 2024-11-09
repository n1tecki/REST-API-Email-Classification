import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import mlflow
import mlflow.sklearn

def train_model(data_path='data/complaints_processed.csv'):
    # Set up MLflow experiment
    mlflow.set_experiment("Customer Complaints Classification")
    
    with mlflow.start_run() as run:
        # Load dataset
        data = pd.read_csv(data_path)
        X = data['narrative']
        y = data['product']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create text processing and modeling pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', LogisticRegression(max_iter=1000))
        ])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Log model parameters and metrics with MLflow
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        accuracy = pipeline.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # Save and register the model
        model_name = "CustomerComplaintsModel"
        model_path = "model/complaints_classifier.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.sklearn.log_model(pipeline, artifact_path="model", registered_model_name=model_name)

        print(f"Model registered with run ID: {run.info.run_id}")

    print("Model trained, saved, and registered.")


train_model()
#if __name__ == "__main__":
#    train_model()
