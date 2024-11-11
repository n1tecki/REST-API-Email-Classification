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
model_loader = ModelLoader(model_uri=f"models:/{config['model']['model_name']}/production")
data_monitor = DataMonitor(config["data"]["data_path"], config["data"]["log_path"])



@app.post("/predict")
def predict_category(complaint: Complaint, api_key: str = Header(...)):
    if not verify_api_key(api_key, hashed_key):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    try:
        prediction = model_loader.predict(complaint.text)
        data_monitor.collect_data(complaint.text, prediction)

        return {"category": prediction}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/model/manage")
def manage_model(version: int, api_key: str = Header(...)):
    if not verify_api_key(api_key, hashed_key):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    try:
        # Get the model URI for a specific version, without using stage
        model_uri = f"models:/CustomerComplaintsModel/{version}"

        global model_loader
        model_loader = ModelLoader(model_uri=model_uri)

        return {"message": f"Model version {version} reloaded for predictions"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# mlflow ui
# uvicorn app.main:app --reload
# curl -X POST "http://127.0.0.1:8000/predict" -H "accept: application/json" -H "api-key: mysecretkey" -H "Content-Type: application/json" -d "{\"text\": \"I have an issue with my loan.\"}"
# curl -X POST "http://127.0.0.1:8000/model/manage" -H "accept: application/json" -H "api_key: mysecretkey" -d '{"version": 2}'

