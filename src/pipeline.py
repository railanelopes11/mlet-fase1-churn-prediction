"""
Construção do Pipeline de churn: pré-processamento (escala + encoding) e modelo.
Mantém o encoding e a escala dentro do Pipeline para que sejam ajustados
apenas com os dados de treino, evitando vazamento de dados.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Limiar de decisão: usado no treino (métricas em models/metrics.json) e na API
LIMIAR_CHURN = 0.5

COLUNAS_NUMERICAS = ['tenure', 'MonthlyCharges', 'TotalCharges']
COLUNAS_JA_BINARIAS = ['SeniorCitizen']  # já vem como 0/1 no CSV
COLUNAS_CATEGORICAS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod',
]


def construir_pipeline(estimador) -> Pipeline:
    """Monta o Pipeline completo, com o pré-processamento seguido do estimador recebido."""
    preprocessador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), COLUNAS_NUMERICAS),
            ('cat', OneHotEncoder(handle_unknown='ignore'), COLUNAS_CATEGORICAS),
            ('bin', 'passthrough', COLUNAS_JA_BINARIAS),
        ],
        sparse_threshold=0,  # saída sempre densa (dataset pequeno)
    )
    return Pipeline([('pre', preprocessador), ('modelo', estimador)])
