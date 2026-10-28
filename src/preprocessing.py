"""
Funções de pré-processamento dos dados de churn.
Replica os passos de limpeza feitos na EDA (EDA + Baseline.ipynb).
O encoding e a escala das features ficam no Pipeline (src/pipeline.py),
para serem ajustados apenas com os dados de treino.
"""

import pandas as pd


def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    """Carrega o CSV bruto do dataset de churn."""
    return pd.read_csv(caminho_csv)


def tratar_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige o tipo da coluna TotalCharges e preenche nulos com 0."""
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas não usadas no modelo (customerID) e converte Churn
    para 0/1, quando presente. As categóricas continuam como texto:
    o OneHotEncoder do Pipeline cuida delas.
    """
    df = df.copy()
    df = df.drop(columns=['customerID'], errors='ignore')

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df
