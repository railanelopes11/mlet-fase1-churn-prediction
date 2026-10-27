"""
Funções de pré-processamento dos dados de churn.
Replica exatamente os passos feitos na EDA (EDA + Baseline.ipynb) e na
preparação dos dados para modelagem (model_comparison.ipynb).
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
    Remove colunas não usadas no modelo, converte Churn para 0/1 e
    faz encoding das variáveis categóricas.

    Se `encoders` for None (modo treino), cria e ajusta um LabelEncoder
    por coluna categórica, retornando-os para reuso posterior.
    Se `encoders` for fornecido (modo inferência), reutiliza os encoders
    já ajustados no treino, sem reaprender categorias.
    """
    df = df.copy()
    df = df.drop(columns=['customerID'])

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df

    categoricas = df.select_dtypes(include=['object', 'str']).columns

    # if encoders is None:
    #     encoders = {}
    #     for col in categoricas:
    #         le = LabelEncoder()
    #         df[col] = le.fit_transform(df[col])
    #         encoders[col] = le
    # else:
    #     for col in categoricas:
    #         df[col] = encoders[col].transform(df[col])

    # return df, encoders