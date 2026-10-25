# Predição de Churn — Telco Customer Churn

Pipeline de Machine Learning para prever quais clientes de uma operadora de telecomunicações têm maior risco de cancelar o serviço (*churn*), permitindo ações de retenção proativas. O modelo final é servido por uma API REST construída com **FastAPI**.

## Sumário

- [Problema de negócio](#problema-de-negócio)
- [Resultados](#resultados)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Rodando a API localmente](#rodando-a-api-localmente)
- [Exemplos de uso da API](#exemplos-de-uso-da-api)
- [Retreinando o modelo](#retreinando-o-modelo)
- [Rodando os testes](#rodando-os-testes)
- [Notebooks](#notebooks)
- [Documentação](#documentação)

## Problema de negócio

A operadora está perdendo clientes em ritmo acelerado. O objetivo é identificar com antecedência os clientes com maior risco de cancelamento para que o time de retenção atue antes da saída (desconto, upgrade, suporte).

Como **deixar de identificar um cliente que vai cancelar (falso negativo)** custa mais caro do que abordar um cliente que não cancelaria (falso positivo), a métrica principal escolhida é o **recall da classe churn**. Detalhes em [`docs/ml_canvas.md`](docs/ml_canvas.md).

**Dataset:** [Telco Customer Churn (IBM, via Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7.043 clientes, 21 colunas, 26,5% de churn.

## Resultados

Modelo campeão: **Regressão Logística com `class_weight='balanced'`** + `StandardScaler`.

| Métrica (conjunto de teste) | Valor |
|---|---|
| **Recall (churn)** | **0,80** |
| Precision (churn) | 0,51 |
| F1-score (churn) | 0,62 |
| Acurácia | 0,75 |
| ROC-AUC | 0,84 |
| Recall em validação cruzada (5-fold) | 0,80 ± 0,025 |

Também foram testados Random Forest, Árvore de Decisão e MLP (com e sem SMOTE). Nenhum superou a regressão logística em recall. Performance completa, limitações e análise de vieses estão no **[Model Card](docs/model_card.md)**.

## Estrutura do projeto

```
mlet-fase1-churn-prediction/
├── data/
│   └── Telco-Customer-Churn.csv     # dataset bruto
├── docs/
│   ├── ml_canvas.md                 # ML Canvas (contexto de negócio e métricas)
│   └── model_card.md                # Model Card (performance, limitações, vieses)
├── models/
│   ├── churn_model.pkl              # modelo treinado (LogisticRegression)
│   ├── scaler.pkl                   # StandardScaler ajustado no treino
│   └── encoders.pkl                 # LabelEncoders das variáveis categóricas
├── notebooks/
│   ├── EDA + Baseline.ipynb         # Etapa 1: análise exploratória e baseline
│   └── model_comparison.ipynb       # Etapa 2: comparação de modelos
├── src/
│   ├── api.py                       # API FastAPI (/health e /predict)
│   ├── predict.py                   # carregamento do modelo e predição
│   ├── preprocessing.py             # funções de pré-processamento
│   └── train_final_model.py         # script de treino do modelo final
├── tests/
│   ├── test_api.py
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

## Pré-requisitos

- **Python 3.11 ou superior** (testado com Python 3.12). As versões fixadas de `pandas` 3.x e `numpy` 2.5 não funcionam no Python 3.10 ou anterior.
- `pip` e `git`

Para conferir sua versão:

```bash
python --version
```

## Instalação

1. **Clone o repositório** e entre na pasta:

   ```bash
   git clone <url-do-repositorio>
   cd mlet-fase1-churn-prediction
   ```

2. **Crie um ambiente virtual:**

   ```bash
   python -m venv venv
   ```

3. **Ative o ambiente virtual:**

   - Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - Windows (CMD):
     ```cmd
     venv\Scripts\activate.bat
     ```
   - Linux / macOS:
     ```bash
     source venv/bin/activate
     ```

4. **Instale as dependências:**

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

> O `requirements.txt` também inclui Jupyter, Matplotlib e Seaborn, usados nos notebooks. Para rodar **apenas a API**, o mínimo necessário é: `pandas`, `scikit-learn`, `joblib`, `fastapi`, `uvicorn` e `pydantic`.

## Rodando a API localmente

Os artefatos do modelo já treinado estão em `models/`, então não é preciso treinar antes de subir a API.

A partir da **raiz do projeto**, com o ambiente virtual ativado:

```bash
uvicorn api:app --app-dir src --reload
```

A API ficará disponível em **http://127.0.0.1:8000**.

| Endpoint | Método | Descrição |
|---|---|---|
| `/health` | GET | Verifica se a API está no ar |
| `/predict` | POST | Recebe uma lista de clientes e retorna a probabilidade de churn de cada um |
| `/docs` | GET | Documentação interativa (Swagger UI), onde dá para testar as requisições pelo navegador |
| `/redoc` | GET | Documentação alternativa (ReDoc) |

Para usar outra porta: `uvicorn api:app --app-dir src --reload --port 8080`.

## Exemplos de uso da API

### Health check

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok"}
```

### Predição

O corpo da requisição é uma **lista** de clientes, com os mesmos campos do CSV original (sem a coluna `Churn`). O campo `customerID` é opcional.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }]'
```

Resposta:

```json
[
  {
    "customerID": "7590-VHVEG",
    "probabilidade_churn": 0.8064,
    "churn_previsto": true,
    "limiar": 0.5
  }
]
```

No **PowerShell**, use `Invoke-RestMethod` ou, mais simples, teste pelo Swagger em http://127.0.0.1:8000/docs.

Exemplo em Python:

```python
import requests

cliente = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}

resposta = requests.post("http://127.0.0.1:8000/predict", json=[cliente])
print(resposta.json())
```

**Valores aceitos nos campos categóricos:**

| Campo | Valores |
|---|---|
| `gender` | `Female`, `Male` |
| `SeniorCitizen` | `0`, `1` |
| `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | `Yes`, `No` |
| `MultipleLines` | `Yes`, `No`, `No phone service` |
| `InternetService` | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | `Yes`, `No`, `No internet service` |
| `Contract` | `Month-to-month`, `One year`, `Two year` |
| `PaymentMethod` | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |

Valores fora dessa lista retornam **HTTP 422** com a descrição do erro.

> A `probabilidade_churn` deve ser lida como um **score de risco** para priorização. Como o modelo foi treinado com classes balanceadas, ela tende a superestimar a probabilidade real (ver [Model Card](docs/model_card.md#6-limitações)).

## Retreinando o modelo

Para regerar os artefatos em `models/` a partir do CSV (mesmo pipeline do notebook `model_comparison.ipynb`):

```bash
python src/train_final_model.py
```

## Rodando os testes

```bash
pytest
```

## Notebooks

Para abrir os notebooks de análise:

```bash
jupyter lab
```

- **`notebooks/EDA + Baseline.ipynb`**: análise exploratória (tratamento de `TotalCharges`, taxas de churn por variável, correlações) e baseline com regressão logística.
- **`notebooks/model_comparison.ipynb`**: comparação entre Regressão Logística, Random Forest, Árvore de Decisão e MLP (com e sem SMOTE), e validação cruzada do modelo campeão.

## Documentação

- [ML Canvas](docs/ml_canvas.md): contexto de negócio, stakeholders, custos de erro e métricas.
- [Model Card](docs/model_card.md): performance, limitações e possíveis vieses do modelo.
