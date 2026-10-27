# Model Card — Modelo de Predição de Churn (Telco)

## 1. Visão geral

| Item | Descrição |
|---|---|
| **Nome** | `churn_pipeline.pkl` — Classificador de churn |
| **Versão** | 2.0.0 (pré-processamento e modelo unificados em um único `Pipeline`) |
| **Tipo** | Classificação binária (1 = cliente cancela, 0 = cliente permanece) |
| **Algoritmo** | Regressão Logística (`scikit-learn`), `class_weight='balanced'`, `max_iter=1000`, `random_state=1109` |
| **Pré-processamento** | Correção de `TotalCharges` (texto → numérico, nulos → 0) e remoção de `customerID` ([`src/preprocessing.py`](../src/preprocessing.py)); dentro do `Pipeline` ([`src/pipeline.py`](../src/pipeline.py)): `StandardScaler` em `tenure`, `MonthlyCharges` e `TotalCharges`, `OneHotEncoder` nas 15 colunas categóricas e `SeniorCitizen` (já 0/1) sem transformação. Encoding e escala são ajustados só com os dados de treino |
| **Artefatos** | `models/churn_pipeline.pkl` (pré-processamento + modelo), `models/metrics.json` (métricas no conjunto de teste) |
| **Código de treino** | [`src/train_final_model.py`](../src/train_final_model.py) |
| **Notebooks** | [`EDA + Baseline.ipynb`](../notebooks/EDA%20+%20Baseline.ipynb), [`model_comparison.ipynb`](../notebooks/model_comparison.ipynb) |
| **Saída** | Probabilidade de churn (0 a 1); classificado como churn quando a probabilidade é ≥ 0,5 |

## 2. Uso pretendido

- **Uso principal:** priorizar clientes para ações **proativas de retenção** (contato, desconto, upgrade, suporte), conforme definido no [ML Canvas](ml_canvas.md).
- **Usuários:** time de retenção/atendimento e diretoria.
- **O modelo apoia uma decisão humana**, não a substitui. A saída é um ranking de risco, não um veredito.

### Usos fora do escopo

- Negar serviço, crédito, aumentar preço ou tratar pior um cliente por ele ser considerado "em risco".
- Aplicar em outras empresas, países ou produtos sem antes retreinar e validar o modelo.
- Explicar causalidade ("o cliente cancela *porque* paga com cheque eletrônico"). O modelo encontra correlações, não causas.

## 3. Dados

- **Fonte:** [Telco Customer Churn (IBM Sample Data, via Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- **Tamanho:** 7.043 clientes, 19 features usadas no modelo (20 colunas originais menos `customerID`) + alvo `Churn`. Após o One-Hot, o modelo recebe 45 colunas.
- **Distribuição do alvo:** 73,5% não cancelaram / 26,5% cancelaram (dados desbalanceados).
- **Divisão:** treino/teste 80/20 estratificado (`random_state=1109`) → 5.634 treino / 1.409 teste (1.035 não-churn, 374 churn).
- **Tratamentos:** 11 registros com `TotalCharges` vazio (todos com `tenure = 0`, clientes sem ciclo de cobrança completo) foram preenchidos com 0.

## 4. Métrica de avaliação

O custo de um **falso negativo** (cliente cancela sem ser abordado) é maior do que o de um **falso positivo** (contato/desconto desnecessário). Por isso a métrica principal é o **recall da classe 1 (churn)**, acompanhado de F1-score e precision. Acurácia **não** é a métrica principal, pois o desbalanceamento a infla.

## 5. Performance

### 5.1 Conjunto de teste (1.409 clientes, limiar 0,5)

| Métrica | Valor |
|---|---|
| **Recall (churn)** | **0,79** |
| Precision (churn) | 0,52 |
| F1-score (churn) | 0,63 |
| Acurácia | 0,75 |
| ROC-AUC | 0,84 |
| PR-AUC (Average Precision) | 0,66 |
| Brier score | 0,166 |

Recall, precision, F1 e ROC-AUC são gravados em `models/metrics.json` a cada execução de `src/train_final_model.py`.

**Matriz de confusão**

|  | Previsto: não churn | Previsto: churn |
|---|---|---|
| **Real: não churn** | 759 (VN) | 276 (FP) |
| **Real: churn** | 77 (FN) | 297 (VP) |

Em termos de negócio: de cada 10 clientes que realmente cancelam, o modelo identifica 8. Em troca, cerca de metade dos clientes sinalizados não cancelaria.

### 5.2 Validação cruzada (5-fold estratificado)

Recall por fold: 0,832 · 0,829 · 0,778 · 0,802 · 0,783 → **média 0,80 ± 0,022**. O resultado é estável e não depende de um split específico. O pré-processamento é reajustado dentro de cada fold (sem vazamento de dados de validação).

### 5.3 Comparação com outros modelos testados

Todos os modelos usam o mesmo pré-processamento (`Pipeline`). Detalhes em [`model_comparison.ipynb`](../notebooks/model_comparison.ipynb).

| Modelo | Recall (churn) | Precision (churn) | F1 (churn) | Acurácia |
|---|---|---|---|---|
| **Regressão Logística (balanced)** ✅ | **0,79** | 0,52 | 0,63 | 0,75 |
| Regressão Logística (sem balanceamento)* | 0,55 | 0,67 | 0,60 | 0,81 |
| Random Forest (balanced) | 0,64 | 0,56 | 0,60 | 0,77 |
| Árvore de Decisão (balanced) | 0,54 | 0,50 | 0,52 | 0,73 |
| MLP (sem balanceamento) | 0,50 | 0,60 | 0,54 | 0,78 |
| MLP (sample_weight balanced) | 0,69 | 0,50 | 0,58 | 0,73 |
| MLP (SMOTE) | 0,57 | 0,53 | 0,55 | 0,75 |

\* Baseline da Etapa 1 ([`EDA + Baseline.ipynb`](../notebooks/EDA%20+%20Baseline.ipynb)), com `LabelEncoder`.

### 5.4 Sensibilidade ao limiar de decisão

| Limiar | Recall | Precision | F1 | % da base sinalizada |
|---|---|---|---|---|
| 0,3 | 0,93 | 0,42 | 0,58 | 58% |
| 0,4 | 0,86 | 0,46 | 0,60 | 50% |
| **0,5 (padrão)** | **0,79** | **0,52** | **0,63** | **41%** |
| 0,6 | 0,72 | 0,56 | 0,63 | 34% |
| 0,7 | 0,59 | 0,63 | 0,61 | 25% |

O limiar pode ser ajustado conforme a capacidade do time de retenção e o custo das ações.

### 5.5 Variáveis mais influentes

Coeficientes da regressão logística após o pré-processamento (numéricas padronizadas, categóricas em One-Hot; positivo = aumenta o risco de churn). Com One-Hot, cada categoria tem o seu próprio coeficiente:

| Variável | Coeficiente | Interpretação |
|---|---|---|
| `tenure` | −1,35 | Quanto mais tempo como cliente, menor o risco |
| `Contract = Two year` | −0,82 | Contrato de 2 anos reduz bastante o risco |
| `TotalCharges` | +0,65 | Correlacionada com `tenure` (0,83) e `MonthlyCharges`; ver limitações |
| `Contract = Month-to-month` | +0,64 | Contrato mensal aumenta o risco |
| `InternetService = DSL` | −0,32 | DSL associado a menor risco |
| `OnlineSecurity = Yes` | −0,28 | Ter segurança online reduz o risco |
| `InternetService = Fiber optic` | +0,28 | Fibra associada a maior risco |
| `SeniorCitizen` | +0,26 | Idosos têm risco maior (ver seção 7) |
| `PaymentMethod = Electronic check` | +0,24 | Cheque eletrônico associado a maior risco |
| `TechSupport = Yes` | −0,21 | Ter suporte técnico reduz o risco |

`gender` tem coeficientes quase iguais para as duas categorias (−0,076 e −0,074): a diferença entre homens e mulheres é de ≈ 0,002, ou seja, praticamente sem influência. Os coeficientes são exibidos no final de [`model_comparison.ipynb`](../notebooks/model_comparison.ipynb).

## 6. Limitações

1. **Precision moderada (0,52):** o modelo foi otimizado para recall, então gera muitos falsos positivos (~41% da base é sinalizada). Isso tem custo operacional e de descontos.
2. **Multicolinearidade:** `TotalCharges` ≈ `tenure × MonthlyCharges`, e o One-Hot gera colunas redundantes (ex.: `No internet service` aparece em várias variáveis). Os coeficientes devem ser interpretados com cuidado (o sinal positivo de `TotalCharges` não significa que "gastar mais no total" aumente o churn por si só).
3. **Probabilidades não calibradas:** com `class_weight='balanced'` as probabilidades ficam infladas para a classe churn (o modelo prevê ~41% de churn contra 26,5% reais). Use a saída como **ranking de risco**, não como probabilidade real.
4. **Dados estáticos e sintéticos:** o dataset da IBM é uma amostra fictícia, de uma única "foto" no tempo, sem sazonalidade, histórico de uso, reclamações ou interações com o atendimento. A performance em dados reais pode ser diferente.
5. **Desempenho desigual entre segmentos:** o modelo praticamente **não detecta churn em contratos de 2 anos** (recall 0,00 em 12 casos no teste), em clientes **sem internet** (recall 0,00 em 25 casos) e é fraco em contratos de 1 ano (0,26) e clientes com mais de 4 anos de casa (0,41). São grupos com pouco churn, mas os cancelamentos que acontecem ali passam despercebidos.
6. **Categorias novas:** valores não vistos no treino (ex.: um novo método de pagamento) não são suportados. A API rejeita essas requisições (HTTP 422); usado direto em Python, o `OneHotEncoder` (`handle_unknown='ignore'`) ignoraria a categoria em silêncio.
7. **Sem monitoramento de drift:** ainda não há acompanhamento de mudança na distribuição dos dados ou queda de performance em produção.

## 7. Possíveis vieses e análise de equidade

Métricas no conjunto de teste por grupo (limiar 0,5). *FPR = taxa de falsos positivos (clientes que não cancelariam, mas foram sinalizados).*

| Grupo | n | Churn real | % sinalizado | Recall | Precision | FPR | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Feminino | 709 | 26,9% | 41,2% | 0,79 | 0,51 | 0,27 | 0,84 |
| Masculino | 700 | 26,1% | 40,1% | 0,80 | 0,52 | 0,26 | 0,85 |
| Não idoso | 1.177 | 23,9% | 35,4% | 0,75 | 0,50 | 0,23 | 0,84 |
| **Idoso** | 232 | 40,1% | **67,2%** | 0,94 | 0,56 | **0,50** | 0,81 |
| Sem parceiro(a) | 689 | 33,5% | 50,4% | 0,81 | 0,54 | 0,35 | 0,82 |
| Com parceiro(a) | 720 | 19,9% | 31,4% | 0,76 | 0,48 | 0,20 | 0,85 |
| Sem dependentes | 971 | 31,8% | 48,9% | 0,83 | 0,54 | 0,33 | 0,83 |
| Com dependentes | 438 | 14,8% | 22,4% | 0,63 | 0,42 | 0,15 | 0,82 |
| Pagamento: cheque eletrônico | 461 | 45,3% | 72,0% | 0,94 | 0,59 | 0,54 | 0,81 |
| Pagamento: automático (cartão/banco) | 629 | 16,7% | 23,5% | 0,61 | 0,43 | 0,16 | 0,81 |

**Principais observações:**

- **Gênero:** desempenho equivalente entre homens e mulheres; a diferença entre os coeficientes de `Female` e `Male` é praticamente zero. Não há evidência de viés de gênero.
- **Idosos (`SeniorCitizen`):** são sinalizados com muito mais frequência (67% vs. 35%) e **metade dos idosos que não cancelariam é marcada como risco** (FPR 0,50 vs. 0,23). Parte disso reflete a taxa real de churn maior (40% vs. 24%), mas significa que idosos receberão mais abordagens de retenção. Se a ação for um benefício (desconto), isso pode ser até positivo; se for uma ação invasiva (ligações insistentes) ou restritiva, configura um tratamento desigual por idade.
- **Situação familiar (`Partner`, `Dependents`):** clientes sozinhos/sem dependentes são mais sinalizados; clientes com dependentes têm recall menor (0,63), ou seja, seus cancelamentos são detectados com menos frequência.
- **Método de pagamento:** cheque eletrônico concentra muitos falsos positivos (FPR 0,54) e 72% desses clientes são sinalizados. Essa variável pode funcionar como **proxy de perfil socioeconômico** ou de nível de bancarização.
- **Atributos sensíveis no modelo:** `gender`, `SeniorCitizen`, `Partner` e `Dependents` são usados diretamente como features. Dependendo do uso (e da LGPD), pode ser necessário removê-los ou justificar seu uso, e verificar se as decisões tomadas com o modelo não discriminam esses grupos.
- **Viés do dataset:** dados sintéticos de uma operadora norte-americana; não representam necessariamente a base de clientes de uma operadora brasileira.

## 8. Recomendações

- Usar o modelo como **ferramenta de priorização**, com revisão humana das ações de retenção.
- Garantir que a ação disparada pelo modelo seja **benéfica ao cliente** (oferta, suporte), nunca punitiva.
- Monitorar periodicamente recall/precision **por segmento** (idade, contrato, pagamento), não só no agregado.
- Próximas melhorias: calibração de probabilidades (`CalibratedClassifierCV`), avaliar remoção de atributos sensíveis, tuning de hiperparâmetros e ajuste do limiar conforme o custo real das ações.
- Retreinar o modelo quando houver dados reais ou drift detectado.
