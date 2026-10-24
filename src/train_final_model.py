"""
Script de treino do modelo final de churn.
Reproduz o pipeline validado na Etapa 2 (model_comparison.ipynb),
usando os módulos de pré-processamento, para gerar os artefatos
finais (modelo, scaler, encoders) de forma reprodutível.
"""

import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from preprocessing import carregar_dados, tratar_total_charges, preparar_features


def main():
    df = carregar_dados('data/Telco-Customer-Churn.csv')
    df = tratar_total_charges(df)
    df_model, encoders = preparar_features(df)

    X = df_model.drop(columns=['Churn'])
    y = df_model['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=1109
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    modelo = LogisticRegression(max_iter=1000, random_state=1109, class_weight='balanced')
    modelo.fit(X_train_scaled, y_train)

    joblib.dump(modelo, 'models/churn_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')

    print("Modelo, scaler e encoders salvos em models/")


if __name__ == '__main__':
    main()