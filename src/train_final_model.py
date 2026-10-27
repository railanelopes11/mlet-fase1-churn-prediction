"""
Script de treino do modelo final de churn.
Reproduz o pipeline validado na Etapa 2 (model_comparison.ipynb),
usando os módulos de pré-processamento e o Pipeline compartilhado
(src/pipeline.py), para gerar de forma reprodutível o artefato final
(models/churn_pipeline.pkl) e as métricas no conjunto de teste
(models/metrics.json).
"""

import json
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from preprocessing import carregar_dados, tratar_total_charges, preparar_features
from pipeline import construir_pipeline

ROOT = Path(__file__).resolve().parent.parent
DIR_DATA = ROOT / 'data'
DIR_MODELS = ROOT / 'models'
SEED = 1109  # para reprodutibilidade
THRESHOLD = 0.5  # para classificação binária

def main():
    df = carregar_dados(DIR_DATA / 'Telco-Customer-Churn.csv')
    df = tratar_total_charges(df)
    df_model = preparar_features(df)

    X = df_model.drop(columns=['Churn'])
    y = df_model['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    
    pipeline = construir_pipeline(
        LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced'))
    pipeline.fit(X_train, y_train)
    
    prob = pipeline.predict_proba(X_test)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)
    
    print(classification_report(y_test, pred))
    metrics = {
        'recall': recall_score(y_test, pred),
        'precision': precision_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'roc_auc': roc_auc_score(y_test, prob),
        'threshold': THRESHOLD,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()})
    
    DIR_MODELS.mkdir(exist_ok=True)
    joblib.dump(pipeline, DIR_MODELS / 'churn_pipeline.pkl')
    (DIR_MODELS / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    print("Pipeline e métricas salvas em", DIR_MODELS)

if __name__ == '__main__':
    main()