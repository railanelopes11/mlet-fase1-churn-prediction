"""
Funções de carregamento do modelo treinado e geração de predições.
"""

import joblib
import pandas as pd


def carregar_modelo(caminho_modelo: str):
    """Carrega o Pipeline de churn treinado (.pkl): pré-processamento + modelo."""
    return joblib.load(caminho_modelo)


def prever_churn(dados: pd.DataFrame, modelo) -> list[float]:
    """
    Recebe um DataFrame já limpo por preparar_features (sem a coluna Churn)
    e retorna a probabilidade de churn (classe 1) para cada linha.
    O encoding e a escala são aplicados pelo próprio Pipeline.
    """
    probabilidades = modelo.predict_proba(dados)[:, 1]
    return probabilidades.tolist()
