from fastapi import FastAPI, Body
from pydantic import BaseModel
import csv
from csv import writer
from fastapi.middleware.cors import CORSMiddleware
import script_model
import os
from fastapi.responses import FileResponse

def readFile(inputFile):
    data = []
    # Opening csv file
    with open(inputFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data.append(row)
    
    return data

def writeFile(outputFile, new_data):
    # Write the initial json object (list of dicts)
    with open(outputFile, mode='a+',encoding='utf8') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(new_data)

inputFile = '../../datasource/Wines.csv'
wines = readFile(inputFile)


app = FastAPI()

# CORS authorization
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

"""
fixed acidity:  float
volatile acidity:  float
citric acid:  float
residual sugar:  float
chlorides:  float
free sulfur dioxide:  float
total sulfur dioxide:  float
density:  float
pH:  float
sulphates:  float
alcohol:  float
quality:  float
Id
"""
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

"""
Permet de réaliser une prédiction en donnant en body les données nécessaires du vin à celle-ci
• La prédiction devra être donnée via une note sur 10 du vin entré.
"""
@app.post("/api/predict/") 
async def predict_quality(wine: Wine):
    quality = script_model.prediction(wine)
    return {"predicted_quality": int(quality)}

"""
Permet de générer une combinaison de données permettant d'identifier le “vin parfait” (probablement
inexistant mais statistiquement possible)
• La prédiction devra fournir les caractéristiques du “vin parfait”
"""
@app.get("/api/predict/")
async def predict_perfect_wine():
    perfect_wine = {0.2,0.4,0.5,0.6,0.7,0.8,0.9,0.4,0.8,0.4}
    return perfect_wine

"""
Permet d'obtenir le modèle sérialisé
"""
@app.get("/api/model/")
async def get_model(model_name='random_forest.joblib'):
    script_model.get_model()
    path = os.path.join('../../model/', model_name)
    
    if os.path.exists(path):
        return FileResponse(path, media_type="text/plain", filename=model_name)

"""
Permet d'obtenir des informations sur le modèle
• Paramètres du modèle
• Métriques de performance du modèle sur jeu de test (dernier entraînement)
• Autres (Dépend de l'algo utilisé)
"""
@app.get("/api/model/description/")
async def get_model():
    model_description = script_model.get_model_information()
    return {"Paramètres du modèle" : model_description["Paramètres du modèle"], "Métriques principales": model_description["Métriques principales"]}

"""
Permet d'enrichir le modèle d'une entrée de donnée supplémentaire
(un vin en plus)
• Une donnée supplémentaire doit avoir le même format que le reste des données.
"""
@app.put("/api/model/")
async def add_wine(new_wine: WineFull = Body()):
    script_model.add_wine(new_wine.__dict__)
    return {"add_wine": "New wine was added"}

"""
Permet de réentrainer le modèle
• Il doit prendre en compte les données rajoutées a posteriori
"""
@app.post("/api/model/retrain/")
async def retrain_model(): #model: str, inputfile: str
    script_model.model_train()
    return 0 
