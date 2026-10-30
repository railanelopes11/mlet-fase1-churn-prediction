import sys
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ_PROJETO / 'src'))

import pytest
from fastapi.testclient import TestClient

from api import CLIENTE_EXEMPLO, SERVICOS_DE_INTERNET, TENURE_MAXIMO, Cliente, app

client = TestClient(app)


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


def test_predict_aceita_total_charges_ausente_com_tenure_zero():
    cliente = {k: v for k, v in CLIENTE_EXEMPLO.items() if k != 'TotalCharges'}
    resposta = client.post('/predict', json={**cliente, 'tenure': 0})
    assert resposta.status_code == 200


@pytest.mark.parametrize(
    'alteracoes',
    [
        {'tenure': 60, 'TotalCharges': None},
        {'tenure': 60, 'TotalCharges': ' '},
        {'tenure': 60, 'TotalCharges': 0},
        {'tenure': TENURE_MAXIMO + 1},
        {'tenure': -1},
        {'PhoneService': 'Yes', 'MultipleLines': 'No phone service'},
        {'PhoneService': 'No', 'MultipleLines': 'Yes'},
        {'InternetService': 'DSL', 'TechSupport': 'No internet service'},
        *[{'InternetService': 'No', campo: 'No'} for campo in SERVICOS_DE_INTERNET],
    ],
)
def test_predict_rejeita_dados_inconsistentes(alteracoes):
    resposta = client.post('/predict', json={**CLIENTE_EXEMPLO, **alteracoes})
    assert resposta.status_code == 422


def test_predict_rejeita_total_charges_ausente_com_tenure_positivo():
    cliente = {k: v for k, v in CLIENTE_EXEMPLO.items() if k != 'TotalCharges'}
    resposta = client.post('/predict', json=cliente)
    assert resposta.status_code == 422


def test_predict_erro_explica_o_problema():
    resposta = client.post('/predict', json={**CLIENTE_EXEMPLO, 'tenure': 60, 'TotalCharges': None})
    assert 'TotalCharges deve ser informado' in resposta.text


def test_predict_aceita_cliente_sem_internet():
    servicos = {campo: 'No internet service' for campo in SERVICOS_DE_INTERNET}
    resposta = client.post('/predict', json={**CLIENTE_EXEMPLO, 'InternetService': 'No', **servicos})
    assert resposta.status_code == 200


def test_validacao_aceita_todas_as_linhas_do_dataset():
    from preprocessing import carregar_dados, tratar_total_charges

    df = tratar_total_charges(carregar_dados(RAIZ_PROJETO / 'data' / 'Telco-Customer-Churn.csv'))
    for linha in df.drop(columns=['Churn']).to_dict(orient='records'):
        Cliente(**linha)


def test_predict_bate_com_modelo_direto():
    from predict import carregar_modelo, prever_churn
    from preprocessing import carregar_dados, preparar_features, tratar_total_charges

    df = tratar_total_charges(carregar_dados(RAIZ_PROJETO / 'data' / 'Telco-Customer-Churn.csv')).head(200)
    features = preparar_features(df)
    esperado = prever_churn(
        features.drop(columns=['Churn']),
        carregar_modelo(RAIZ_PROJETO / 'models' / 'churn_pipeline.pkl'),
    )

    payload = df.drop(columns=['Churn']).to_dict(orient='records')
    resposta = client.post('/predict', json=payload)
    assert resposta.status_code == 200
    obtido = [p['probabilidade_churn'] for p in resposta.json()]
    assert all(abs(o - e) < 1e-4 for o, e in zip(obtido, esperado))
