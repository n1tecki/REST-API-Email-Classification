import pandas as pd
import mlflow
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection


class DataMonitor:
    def __init__(self, training_data: str):
        self.incoming_data = []
        self.predictions = []
        self.request_count = 0
        self.training_data = training_data


    def collect_data(self, text, prediction):
        self.incoming_data.append({"text": text})
        self.predictions.append({"prediction": prediction})
        self.request_count += 1


    def analyze_data_drift(self):
        if self.request_count % 10 != 0:
            return

        df_current = pd.DataFrame(self.incoming_data)
        df_reference = pd.read_csv(self.training_data)

        #data_drift_profile = Profile(sections=[DataDriftProfileSection()])
        #data_drift_profile.calculate(df_reference, df_current)

        dashboard = Dashboard(tabs=[DataDriftTab()])
        dashboard.calculate(df_reference, df_current)

        report_filename = "data_drift_report.html"
        dashboard.save(report_filename)
        mlflow.log_artifact(report_filename)
        #drift_summary = data_drift_profile.json()
        #mlflow.log_metric("drift_score", drift_summary["data_drift"]["data"]["metrics"]["drift_score"])



