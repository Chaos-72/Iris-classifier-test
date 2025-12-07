from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import pickle
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path


MODEL_PATH = Path("model/iris_model.pkl")
FRONTEND_INDEX = Path("static/index.html")

print("==================MODEL_PATH: ", MODEL_PATH)
print("==================FRONTEND_INDEX: ", FRONTEND_INDEX)

app = FastAPI(title="Iris Classifier API")

# CORS 

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# server static files from ./static/index.html, css, js

app.mount("/static", StaticFiles(directory='static'), name="static")

# data model for request: 4 numeric features (sepal_length, sepal_width, peta;_length, petal_width)

class Features(BaseModel):
    features: conlist(float, min_length=4, max_length=4)


model_data = {}

@app.on_event("startup")
def load_model():
    global model_data
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file {MODEL_PATH} not found . Run the model.py first")
    
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)

    if "model" not in model_data or 'target_names' not in model_data:
        raise RuntimeError("Pickle file missting except keys 'model' and 'target_name'.")
    
@app.get('/')
def index():
    return FileResponse(FRONTEND_INDEX)

@app.post('/predict')
def predict(payload: Features):
    features = payload.features

    # ensure Length is 4 (pydantic already validates)
    try:
        model = model_data['model']
        target_names = model_data['target_names']

        # model excepts 2D array
        pred = model.predict([features])[0]
        proba = None

        # if the model supports predict_proba, return probabilities
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([features])[0].tolist()
        return {"prediction_index": int(pred), "prediction": str(target_names[int(pred)]), "probabilties": proba}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))