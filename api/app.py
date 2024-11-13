from fastapi import FastAPI, HTTPException, Header, Depends
from api.models import ModelLoader, Complaint
from api.utils.general_utils import load_config, get_hashed_key, verify_api_key
from api.monitoring import DataMonitor
from fastapi.security import OAuth2PasswordBearer




app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
config = load_config()
model_loader = ModelLoader(config["model"]["model_name"])
data_monitor = DataMonitor()



def validate_token(token: str = Depends(oauth2_scheme)):
    valid_hashed_token = config["security"]["hashed_token"]
    if not verify_api_key(token, valid_hashed_token):
        raise HTTPException(status_code=401, detail="Invalid or expired token")



@app.post("/predict")
def predict_category(complaint: Complaint, token: str = Depends(validate_token)):
    try:
        prediction = model_loader.predict(complaint.text)
        data_monitor.collect_data(complaint.text, prediction)
        return {"category": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/model_update")
def manage_model(api_key: str = Depends(validate_token)):
    try:
        global model_loader
        model_loader = ModelLoader(config["model"]["model_name"])
        return {"message": f"Model version {model_loader.model_version} loaded for predictions"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




