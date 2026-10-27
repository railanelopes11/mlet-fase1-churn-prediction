import sys
sys.path.append('src')

import pandas as pd
from preprocessing import tratar_total_charges, preparar_features


def test_tratar_total_charges_converte_para_numerico():
    df = pd.DataFrame({'TotalCharges': ['29.85', ' ', '108.15']})
    resultado = tratar_total_charges(df)
    assert resultado['TotalCharges'].dtype == float
    assert resultado['TotalCharges'].isnull().sum() == 0


def test_preparar_features_remove_customer_id():
    df = pd.DataFrame({
        'customerID': ['0001-ABC'],
        'gender': ['Female'],
        'Churn': ['Yes'],
    })
    resultado = preparar_features(df)
    assert 'customerID' not in resultado.columns
    assert resultado['Churn'].iloc[0] == 1


def test_preparar_features_mantem_categoricas_como_texto():
    df = pd.DataFrame({
        'customerID': ['0001-ABC'],
        'Contract': ['Month-to-month'],
    })
    resultado = preparar_features(df)
    assert resultado['Contract'].iloc[0] == 'Month-to-month'
