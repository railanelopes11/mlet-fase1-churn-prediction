# Predição de Churn — Telco Customer Churn

Pipeline de Machine Learning para prever quais clientes de uma operadora de telecomunicações têm maior risco de cancelar o serviço (*churn*), permitindo ações de retenção proativas. O modelo final é servido por uma API REST construída com **FastAPI**.

## Sumário

- [Problema de negócio](#problema-de-negócio)
- [Resultados](#resultados)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Sobre a API](#sobre-a-api)
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

Também foram testados Random Forest, Árvore de Decisão e MLP (sem balanceamento, com sample_weight e com SMOTE). Nenhum superou a regressão logística em recall. Performance completa, limitações e análise de vieses estão no **[Model Card](docs/model_card.md)**.

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

## Sobre a API

### O que ela faz

A API disponibiliza o modelo de churn treinado como um **serviço web**. Em vez de abrir um notebook e rodar o modelo à mão, qualquer sistema (CRM, planilha, painel, script do time de retenção) pode enviar os dados de um cliente via HTTP e receber na hora:

- a **probabilidade de churn**, um score de risco de 0 a 1;
- uma **classificação** (`churn_previsto`: `true`/`false`), dizendo se o cliente deve ser tratado como em risco de cancelar.

### Para que serve (caso de uso)

1. O time de retenção exporta a lista de clientes ativos.
2. Envia essa lista para o `POST /predict`.
3. A API devolve o score de risco de cada cliente.
4. O time ordena a lista pelo score e prioriza o contato (desconto, upgrade, suporte) com quem tem mais risco, **antes** do cancelamento acontecer.

Isso concretiza a decisão de negócio descrita no [ML Canvas](docs/ml_canvas.md): agir proativamente, e não só depois que o cliente já saiu.

### Como funciona por dentro

```
Requisição JSON (dados brutos do cliente)
        │
        ▼
1. Validação (Pydantic)  ──► campo faltando, tipo errado ou categoria inválida → HTTP 422
        │
        ▼
2. Pré-processamento (src/preprocessing.py)
   • TotalCharges vazio → 0
   • remove customerID
   • converte categorias em números com os LabelEncoders do treino (models/encoders.pkl)
        │
        ▼
3. Padronização com o StandardScaler do treino (models/scaler.pkl)
        │
        ▼
4. Regressão Logística (models/churn_model.pkl) → probabilidade de churn
        │
        ▼
5. Aplica o limiar 0,5 → churn_previsto = probabilidade ≥ 0,5
        │
        ▼
Resposta JSON
```

Os artefatos do modelo são carregados **uma única vez**, quando a API sobe, e reaproveitados em todas as requisições. O pré-processamento usa exatamente as mesmas funções e os mesmos encoders do treino, então a predição da API é idêntica à do modelo rodado direto em Python. Isso foi conferido nas 7.043 linhas do dataset e é coberto pelo teste `test_predict_bate_com_modelo_direto`.

### Endpoints

| Endpoint | Método | O que faz |
|---|---|---|
| `/health` | GET | Retorna `{"status": "ok"}` se a API está no ar. Útil para monitoramento. |
| `/predict` | POST | Recebe **um cliente** (objeto JSON) ou **vários** (lista) e retorna a predição de cada um |
| `/docs` | GET | Swagger UI: documentação interativa, onde dá para testar o `/predict` pelo navegador |
| `/redoc` | GET | Documentação alternativa (ReDoc) |

### Campos de entrada

São os mesmos campos do CSV original, sem a coluna `Churn` (que é o que queremos prever).

| Campo | Tipo | Significado |
|---|---|---|
| `customerID` | texto (opcional) | Identificador do cliente. Não é usado pelo modelo, só devolvido na resposta para você saber de quem é cada predição |
| `gender` | texto | Gênero do cliente |
| `SeniorCitizen` | 0 ou 1 | 1 se o cliente é idoso |
| `Partner` | texto | Se tem parceiro(a) |
| `Dependents` | texto | Se tem dependentes |
| `tenure` | inteiro ≥ 0 | Meses como cliente da operadora |
| `PhoneService` | texto | Se tem serviço de telefone |
| `MultipleLines` | texto | Se tem múltiplas linhas |
| `InternetService` | texto | Tipo de internet (DSL, fibra ou nenhuma) |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` | texto | Serviços adicionais contratados |
| `StreamingTV`, `StreamingMovies` | texto | Se assina streaming de TV/filmes |
| `Contract` | texto | Tipo de contrato (mensal, 1 ano, 2 anos) |
| `PaperlessBilling` | texto | Se usa fatura digital |
| `PaymentMethod` | texto | Forma de pagamento |
| `MonthlyCharges` | número ≥ 0 | Valor da mensalidade |
| `TotalCharges` | número ≥ 0 ou vazio | Total já cobrado. Vazio/nulo vira 0 (clientes recém-chegados) |

Os valores aceitos em cada campo categórico estão na tabela da seção [Exemplos de uso da API](#exemplos-de-uso-da-api).

### Campos de saída

| Campo | Significado |
|---|---|
| `customerID` | O mesmo ID enviado na requisição (ou `"desconhecido"` se não foi enviado) |
| `probabilidade_churn` | Score de risco de cancelamento, de 0 a 1. Quanto maior, maior o risco |
| `churn_previsto` | `true` se `probabilidade_churn ≥ limiar`, ou seja, o cliente deve ser priorizado para retenção |
| `limiar` | Limiar de decisão usado (0,5) |

### Como interpretar o resultado

- Com o limiar 0,5, o modelo identifica **cerca de 80% dos clientes que realmente cancelam** (recall 0,80). Em troca, por volta de metade dos clientes marcados como `churn_previsto: true` não cancelaria (precision 0,51). Esse equilíbrio foi escolhido de propósito, porque perder um cliente sem agir custa mais do que fazer um contato a mais.
- Use `probabilidade_churn` principalmente para **ordenar** os clientes por risco. Como o modelo foi treinado com classes balanceadas, o valor tende a superestimar a probabilidade real de cancelamento.
- O limiar fica na constante `LIMIAR_CHURN` em `src/api.py`. Diminuir o limiar pega mais clientes que cancelariam, mas aumenta os alarmes falsos. Aumentar o limiar faz o contrário. A tabela de trade-off está no [Model Card](docs/model_card.md#54-sensibilidade-ao-limiar-de-decisão).
- O resultado deve **apoiar** a decisão do time de retenção, não substituí-la. Limitações e vieses conhecidos (por exemplo, idosos são sinalizados com mais frequência) estão no [Model Card](docs/model_card.md#7-possíveis-vieses-e-análise-de-equidade).

### Códigos de resposta

| Código | Quando acontece |
|---|---|
| `200` | Predição feita com sucesso |
| `422` | Requisição inválida: campo obrigatório faltando, tipo errado, valor fora das categorias aceitas, número negativo ou lista vazia. A resposta explica qual campo está com problema |

## Rodando a API localmente

Os artefatos do modelo já treinado estão em `models/`, então não é preciso treinar antes de subir a API.

A partir da **raiz do projeto**, com o ambiente virtual ativado:

```bash
uvicorn api:app --app-dir src --reload
```

A API ficará disponível em **http://127.0.0.1:8000**. Abra **http://127.0.0.1:8000/docs** no navegador para testar pelo Swagger (veja os endpoints em [Sobre a API](#sobre-a-api)).

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

O corpo da requisição pode ser **um cliente** (objeto JSON, a resposta vem como objeto) ou **uma lista de clientes** (a resposta vem como lista), com os mesmos campos do CSV original (sem a coluna `Churn`). O campo `customerID` é opcional. `TotalCharges` também pode vir vazio ou nulo (clientes recém-chegados) e é tratado como 0, assim como no treino.

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

# um cliente -> resposta é um objeto
print(requests.post("http://127.0.0.1:8000/predict", json=cliente).json())

# vários clientes -> resposta é uma lista
print(requests.post("http://127.0.0.1:8000/predict", json=[cliente, cliente]).json())
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

A partir da raiz do projeto, com o ambiente virtual ativado:

```bash
pytest
```

> Rode os testes com o Python do ambiente virtual onde o `requirements.txt` foi instalado. Se aparecer `RuntimeError: The starlette.testclient module requires the httpx2 package to be installed`, as dependências foram instaladas em outro interpretador (por exemplo, no Python global). Ative o `venv` e rode `pip install -r requirements.txt` de novo. No VS Code, escolha o interpretador do `venv` em **Python: Select Interpreter**.

## Notebooks

Para abrir os notebooks de análise:

```bash
jupyter lab
```

- **`notebooks/EDA + Baseline.ipynb`**: análise exploratória (tratamento de `TotalCharges`, taxas de churn por variável, correlações) e baseline com regressão logística.
- **`notebooks/model_comparison.ipynb`**: comparação entre Regressão Logística, Random Forest, Árvore de Decisão e MLP (sem balanceamento, com sample_weight e com SMOTE), e validação cruzada do modelo campeão.

## Documentação

- [ML Canvas](docs/ml_canvas.md): contexto de negócio, stakeholders, custos de erro e métricas.
- [Model Card](docs/model_card.md): performance, limitações e possíveis vieses do modelo.
