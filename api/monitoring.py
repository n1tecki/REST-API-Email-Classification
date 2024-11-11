import pandas as pd
import mlflow
from evidently import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from app.util import process
import csv



class DataMonitor:
    def __init__(self, training_data: str, output_csv: str):
        self.buffer = []
        self.buffer_size = 10
        self.request_count = 0
        self.training_data = training_data
        self.output_csv = output_csv
        with open(self.output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['narrative', 'product'])
    

    def collect_data(self, text, prediction):
        self.request_count += 1
        self.buffer.append([text, prediction])
        
        if len(self.buffer) >= self.buffer_size:
            self.flush_buffer()
            self.analyze_data_drift()


    def flush_buffer(self):
        with open(self.output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.buffer)
        self.buffer = []


    def analyze_data_drift(self):
        df_current = process(self.output_csv)
        df_reference = process(self.training_data)

        dashboard = Dashboard(tabs=[DataDriftTab()])
        dashboard.calculate(df_reference, df_current)

        report_filename = "data_drift_report.html"
        dashboard.save(report_filename)
        mlflow.log_artifact(report_filename)



