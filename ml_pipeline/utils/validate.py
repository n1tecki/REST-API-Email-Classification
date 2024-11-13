from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow

class Validator:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=1),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=1),
            "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=1)
        }

        class_report = classification_report(y_test, y_pred)
        mlflow.log_text(class_report, "classification_report.txt")

        return metrics
