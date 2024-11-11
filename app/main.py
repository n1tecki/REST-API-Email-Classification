from fastapi import FastAPI, HTTPException, Header
import mlflow
from mlflow.tracking import MlflowClient
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from app.objects import ModelLoader, Complaint
from app.utils import load_config, get_hashed_key, verify_api_key
from app.monitoring import DataMonitor


app = FastAPI()


config = load_config()
SECRET_API_KEY = config["security"]["api_key"]
hashed_key = get_hashed_key(SECRET_API_KEY)
model_loader = ModelLoader(model_uri=config["model"]["model_path"])
data_monitor = DataMonitor(config["data"]["data_path"])



@app.post("/predict")
def predict_category(complaint: Complaint, api_key: str = Header(...)):
    if not verify_api_key(api_key, hashed_key):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    try:
        prediction = model_loader.predict(complaint.text)

        mlflow.log_param("model_version", model_loader.model_version)
        mlflow.log_param("input_text_length", len(complaint.text))
        mlflow.log_param("predicted_category", prediction)
        data_monitor.collect_data(complaint.text, prediction)
        data_monitor.analyze_data_drift()

        return {"category": prediction}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/model/manage")
def manage_model(version: int, stage: str, api_key: str = Header(...)):
    if not verify_api_key(api_key, hashed_key):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    try:
        #client = MlflowClient()
        #client.transition_model_version_stage(name="CustomerComplaintsModel", version=version, stage=stage)
        
        global model_loader
        model_uri = f"models:/CustomerComplaintsModel/{version}@{stage}"
        model_loader = ModelLoader(model_uri=model_uri)

        return {"message": f"Model version {version} transitioned to {stage} and reloaded for predictions"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
