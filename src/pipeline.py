from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUNAS_NUMERICAS = ['tenure', 'MonthlyCharges', 'TotalCharges']
COLUNAS_JA_BINARIAS = ['SeniorCitizen']  # já vem como 0/1 no CSV
COLUNAS_CATEGORICAS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod',
]

def construir_pipeline(estimador) -> Pipeline:
    preprocessador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), COLUNAS_NUMERICAS),
            ('cat', OneHotEncoder(handle_unknown='ignore'), COLUNAS_CATEGORICAS),
            ('bin', 'passthrough', COLUNAS_JA_BINARIAS),
        ],
        sparse_threshold=0,  # saída sempre densa (dataset pequeno)
    )
    return Pipeline([('pre', preprocessador), ('modelo', estimador)])
