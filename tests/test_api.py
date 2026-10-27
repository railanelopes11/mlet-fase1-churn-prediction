import sys
sys.path.append('src')

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

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


def test_health():
    resposta = client.get('/health')
    assert resposta.status_code == 200
    assert resposta.json() == {'status': 'ok'}


def test_predict_retorna_probabilidade():
    resposta = client.post('/predict', json=[CLIENTE_EXEMPLO])
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert len(corpo) == 1
    assert 0.0 <= corpo[0]['probabilidade_churn'] <= 1.0
    assert corpo[0]['customerID'] == '7590-VHVEG'


def test_predict_rejeita_categoria_invalida():
    cliente = {**CLIENTE_EXEMPLO, 'Contract': 'Three year'}
    resposta = client.post('/predict', json=[cliente])
    assert resposta.status_code == 422


def test_predict_aceita_objeto_unico():
    resposta = client.post('/predict', json=CLIENTE_EXEMPLO)
    assert resposta.status_code == 200
    assert resposta.json()['customerID'] == '7590-VHVEG'


def test_predict_aceita_total_charges_vazio():
    cliente = {**CLIENTE_EXEMPLO, 'tenure': 0, 'TotalCharges': ' '}
    resposta = client.post('/predict', json=[cliente])
    assert resposta.status_code == 200


def test_predict_bate_com_modelo_direto():
    from predict import carregar_modelo, prever_churn
    from preprocessing import carregar_dados, preparar_features, tratar_total_charges

    df = tratar_total_charges(carregar_dados('data/Telco-Customer-Churn.csv')).head(200)
    features = preparar_features(df)
    esperado = prever_churn(
        features.drop(columns=['Churn']),
        carregar_modelo('models/churn_pipeline.pkl'),
    )

    payload = df.drop(columns=['Churn']).to_dict(orient='records')
    resposta = client.post('/predict', json=payload)
    assert resposta.status_code == 200
    obtido = [p['probabilidade_churn'] for p in resposta.json()]
    assert all(abs(o - e) < 1e-4 for o, e in zip(obtido, esperado))
