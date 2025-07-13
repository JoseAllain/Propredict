from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import xgboost

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = joblib.load("pipeline_modelo_completo.pkl")

class InmuebleInput(BaseModel):
    area_constr: float
    area_total: float
    dormitorios: int
    banos: int
    cocheras: int
    antiguedad: int
    tipo: str
    estado: str
    provincia: str
    distrito: str
    piscina: bool
    gimnasio: bool
    sauna: bool
    jacuzzi: bool
    sotano: bool
    closet: bool

@app.post("/predecir")
def predecir_precio(datos: InmuebleInput):
    df = pd.DataFrame([{
        "Area_constr": datos.area_constr,
        "Area_total": datos.area_total,
        "Dormitorios": datos.dormitorios,
        "NroBanios": datos.banos,
        "Cocheras": datos.cocheras,
        "Antiguedad": datos.antiguedad,
        "Tipo": datos.tipo,
        "Estado de Inmueble": datos.estado,
        "Provincia": datos.provincia,
        "Distrito": datos.distrito,
        "Piscina": int(datos.piscina),
        "Gimnasio": int(datos.gimnasio),
        "Sauna": int(datos.sauna),
        "Jacuzzi": int(datos.jacuzzi),
        "Sotano": int(datos.sotano),
        "Walking Closet": int(datos.closet)
    }])

    df["Area_constr_m2"] = df["Area_constr"]
    df["Area_total_m2"] = df["Area_total"]
    df["Precio_M2"] = df["Area_constr"] / df["Area_total"].replace(0, np.nan)
    df["Ratio_Areas"] = df["Area_constr"] / df["Area_total"].replace(0, np.nan)
    df["Total_Habitaciones"] = df["Dormitorios"] + df["NroBanios"]
    df["Indice_Premium"] = (
        df["Piscina"]
        + df["Gimnasio"]
        + df["Sauna"]
        + df["Jacuzzi"]
        + df["Sotano"]
        + df["Walking Closet"]
    )

    df["Precio_M2"] = df["Precio_M2"].fillna(0)
    df["Ratio_Areas"] = df["Ratio_Areas"].fillna(0)

    pred = pipeline.predict(df)[0]
    pred = np.expm1(pred)



    return {"precio_estimado": round(float(pred), 2)}
