from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from objects import ModelLoader, Complaint
from utils import load_config, get_hashed_key, verify_api_key

app = FastAPI()


# Initialize model and preparing hash key
config = load_config()
SECRET_API_KEY = config["security"]["api_key"]
hashed_key = get_hashed_key(SECRET_API_KEY)
model_loader = ModelLoader(model_uri=config["model"]["model_path"])


@app.post("/predict")
def predict_category(complaint: Complaint, api_key: str = Header(...)):
    # Verify API Key
    if not verify_api_key(api_key, hashed_key):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    try:
        prediction = model_loader.predict(complaint.text)
        return {"category": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
