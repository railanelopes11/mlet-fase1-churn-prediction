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
    """
    df = df.copy()
    df = df.drop(columns=['customerID'])

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    categoricas = df.select_dtypes(include=['object', 'str']).columns
    le = LabelEncoder()
    for col in categoricas:
        df[col] = le.fit_transform(df[col])

    return df