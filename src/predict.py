"""
Funções de carregamento do modelo treinado e geração de predições.
"""

import joblib
import pandas as pd


def carregar_modelo(caminho_modelo: str):
    """Carrega o modelo de churn treinado (.pkl)."""
    return joblib.load(caminho_modelo)


def carregar_scaler(caminho_scaler: str):
    """Carrega o StandardScaler ajustado no treino (.pkl)."""
    return joblib.load(caminho_scaler)


def prever_churn(dados: pd.DataFrame, modelo, scaler) -> list[float]:
    """
    Recebe um DataFrame já pré-processado (sem a coluna Churn),
    escala as features e retorna a probabilidade de churn (classe 1)
    para cada linha.
    """
    dados_escalados = scaler.transform(dados)
    probabilidades = modelo.predict_proba(dados_escalados)[:, 1]
    return probabilidades.tolist()