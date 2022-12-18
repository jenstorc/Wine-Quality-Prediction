from fastapi import FastAPI, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import script_model
import os
from fastapi.responses import FileResponse

app = FastAPI(title="Wine Prediction", contact={"email 1": "lebroneclo@cy-tech.fr","email 2":"storchijen@cy-tech.fr"})

#CORS authorization
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Wine(BaseModel):
    fixed_acidity:  float
    volatile_acidity:  float
    citric_acid:  float
    residual_sugar:  float
    chlorides:  float
    free_sulfur_dioxide:  float
    total_sulfur_dioxide:  float
    density:  float
    pH:  float
    sulphates:  float
    alcohol:  float

class WineFull(Wine):
    quality: float
    id: int 


@app.post("/api/predict/") 
async def predict_quality(wine: Wine, quality:int = None):
    """Allows to realize a prediction by giving in body the necessary data of the wine

    Returns:
        _dict_: Quality of the wine
    """
    quality = script_model.prediction(wine)
    return {"predicted_quality": int(quality["prediction"])}


@app.get("/api/predict/")
async def predict_perfect_wine():
    """Generates a combination of data to identify the "perfect wine" (probably
    non-existent but statistically possible)

    Returns:
        _dict_: perfect wine
    """
    perfect_wine = script_model.find_perfect_wine()
    return perfect_wine


@app.get("/api/model/")
async def get_model(model_name='random_forest.joblib'):
    """Allows to obtain the serialized model and download it

    Returns:
        File: model to download
    """
    script_model.get_model()
    path = os.path.join('../../model/', model_name)
    
    if os.path.exists(path):
        return FileResponse(path, media_type="text/plain", filename=model_name)

    
@app.get("/api/model/description/")
async def get_model():
    """Obtain informations about the model

    Returns:
        _dict_: Model parameters, Metric
    """
    model_description = script_model.get_model_information()
    return {"param_model" : model_description["param_model"], "main_metrics": model_description["main_metrics"]}


    
@app.put("/api/model/")
async def add_wine(new_wine: WineFull = Body()):
    """Enables to enrich the model with an additional data input (one more wine)

    Returns:
        _string_: Message
    """
    script_model.add_wine(new_wine)
    return {"add_wine": "New wine was added"}

 
@app.post("/api/model/retrain/")
async def retrain_model(): 
    """Allows to re-train the model

    Returns:
        _string_: Message
    """
    script_model.model_train()
    return {"model_trained": "model retrained"}
