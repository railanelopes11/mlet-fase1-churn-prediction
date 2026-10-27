"""
API REST (FastAPI) para servir o modelo de churn treinado.

Execução local (a partir da raiz do projeto):
    uvicorn api:app --app-dir src --reload
"""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from predict import carregar_modelo, carregar_scaler, prever_churn
from preprocessing import preparar_features, tratar_total_charges

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
DIR_MODELOS = RAIZ_PROJETO / 'models'
LIMIAR_CHURN = 0.5

modelo = carregar_modelo(DIR_MODELOS / 'churn_pipeline.pkl')
# scaler = carregar_scaler(DIR_MODELOS / 'scaler.pkl')
# encoders = joblib.load(DIR_MODELOS / 'encoders.pkl')

SimNao = Literal['Yes', 'No']
SimNaoSemInternet = Literal['Yes', 'No', 'No internet service']

CLIENTE_EXEMPLO = {
    'customerID': '7590-VHVEG',
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85,
}


class Cliente(BaseModel):
    """Dados brutos de um cliente, no mesmo formato do CSV original (sem a coluna Churn)."""

    customerID: str = Field(default='desconhecido', examples=['7590-VHVEG'])
    gender: Literal['Female', 'Male']
    SeniorCitizen: Literal[0, 1]
    Partner: SimNao
    Dependents: SimNao
    tenure: int = Field(ge=0, examples=[1])
    PhoneService: SimNao
    MultipleLines: Literal['Yes', 'No', 'No phone service']
    InternetService: Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity: SimNaoSemInternet
    OnlineBackup: SimNaoSemInternet
    DeviceProtection: SimNaoSemInternet
    TechSupport: SimNaoSemInternet
    StreamingTV: SimNaoSemInternet
    StreamingMovies: SimNaoSemInternet
    Contract: Literal['Month-to-month', 'One year', 'Two year']
    PaperlessBilling: SimNao
    PaymentMethod: Literal[
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)',
    ]
    MonthlyCharges: float = Field(ge=0, examples=[29.85])
    TotalCharges: float | None = Field(
        default=None,
        ge=0,
        examples=[29.85],
        description='Vazio/nulo para clientes sem ciclo de cobrança completo (tratado como 0).',
    )

    model_config = ConfigDict(json_schema_extra={'examples': [CLIENTE_EXEMPLO]})

    @field_validator('TotalCharges', mode='before')
    @classmethod
    def total_charges_vazio_vira_nulo(cls, valor):
        if isinstance(valor, str) and not valor.strip():
            return None
        return valor


class Predicao(BaseModel):
    customerID: str
    probabilidade_churn: float
    churn_previsto: bool
    limiar: float


app = FastAPI(
    title='API de Predição de Churn',
    description='Retorna a probabilidade de cancelamento de um cliente de telecom.',
    version='1.0.0',
)


@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}


@app.post('/predict', response_model=Predicao | list[Predicao])
def predict(clientes: Cliente | list[Cliente]) -> Predicao | list[Predicao]:
    """Aceita um único cliente (objeto JSON) ou vários (lista de objetos)."""
    if isinstance(clientes, Cliente):
        return _prever([clientes])[0]
    if not clientes:
        raise HTTPException(status_code=422, detail='Envie ao menos um cliente.')
    return _prever(clientes)


def _prever(clientes: list[Cliente]) -> list[Predicao]:
    df = pd.DataFrame([c.model_dump() for c in clientes])
    df = tratar_total_charges(df)
    features, _ = preparar_features(df)
    probabilidades = prever_churn(features, modelo)

    return [
        Predicao(
            customerID=cliente.customerID,
            probabilidade_churn=round(prob, 4),
            churn_previsto=prob >= LIMIAR_CHURN,
            limiar=LIMIAR_CHURN,
        )
        for cliente, prob in zip(clientes, probabilidades)
    ]
