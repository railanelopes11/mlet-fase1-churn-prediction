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
| **Notebooks** | [`EDA + Baseline.ipynb`](../notebooks/EDA%20+%20Baseline.ipynb), [`model_comparison.ipynb`](../notebooks/model_comparison.ipynb), [`model_evaluation.ipynb`](../notebooks/model_evaluation.ipynb) |
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

Essas métricas são gravadas em `models/metrics.json` a cada execução de `src/train_final_model.py`. Todos os números desta seção e das seções 5.4, 5.5, 6 e 7 são gerados pelo notebook [`model_evaluation.ipynb`](../notebooks/model_evaluation.ipynb), que avalia o modelo salvo no mesmo conjunto de teste.

**Matriz de confusão**

|  | Previsto: não churn | Previsto: churn |
|---|---|---|
| **Real: não churn** | 759 (VN) | 276 (FP) |
| **Real: churn** | 77 (FN) | 297 (VP) |

Em termos de negócio: de cada 10 clientes que realmente cancelam, o modelo identifica 8. Em troca, cerca de metade dos clientes sinalizados não cancelaria.

### 5.2 Validação cruzada (5-fold estratificado)

Feita **apenas no conjunto de treino** (5.634 clientes), com os mesmos folds (`random_state=1109`) para todos os modelos, deixando o conjunto de teste reservado para a avaliação final. O pré-processamento é reajustado dentro de cada fold (sem vazamento de dados de validação).

Campeão: recall **0,805 ± 0,027**, F1 0,628 ± 0,020, ROC-AUC 0,845 ± 0,013, PR-AUC 0,651 ± 0,017. O resultado é estável e não depende de um split específico.

### 5.3 Comparação com outros modelos testados

Todos os modelos usam o mesmo pré-processamento (`Pipeline`) e foram avaliados nos mesmos 5 folds. Recall, precision e F1 usam o limiar 0,5. ROC-AUC e PR-AUC não dependem de limiar e medem a capacidade de **ordenar** os clientes por risco, o que torna a comparação justa entre modelos balanceados de formas diferentes. Detalhes e curvas ROC/PR em [`model_comparison.ipynb`](../notebooks/model_comparison.ipynb); resultados gravados em `models/cv_results.csv`.

| Modelo (CV, média ± desvio) | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Regressão Logística (balanced)** ✅ | **0,805 ± 0,027** | 0,515 ± 0,026 | **0,628 ± 0,020** | **0,845 ± 0,013** | 0,651 ± 0,017 |
| Regressão Logística (sem balanceamento) | 0,547 ± 0,022 | 0,640 ± 0,027 | 0,590 ± 0,018 | **0,845 ± 0,013** | **0,654 ± 0,016** |
| Random Forest (balanced) | 0,637 ± 0,012 | 0,568 ± 0,021 | 0,600 ± 0,011 | 0,826 ± 0,009 | 0,618 ± 0,023 |
| Árvore de Decisão (balanced) | 0,492 ± 0,013 | 0,506 ± 0,031 | 0,498 ± 0,013 | 0,658 ± 0,010 | 0,383 ± 0,014 |
| MLP (sem balanceamento) | 0,512 ± 0,045 | 0,581 ± 0,034 | 0,542 ± 0,026 | 0,797 ± 0,013 | 0,563 ± 0,028 |
| MLP (sample_weight balanced) | 0,644 ± 0,041 | 0,510 ± 0,015 | 0,568 ± 0,011 | 0,794 ± 0,012 | 0,543 ± 0,012 |
| MLP (SMOTE) | 0,599 ± 0,019 | 0,507 ± 0,017 | 0,549 ± 0,011 | 0,779 ± 0,009 | 0,524 ± 0,010 |

A tabela acima usa os hiperparâmetros padrão do scikit-learn. Com eles, a Regressão Logística balanceada vence Random Forest, Árvore e as três variações de MLP em todos os 5 folds em recall e ROC-AUC. As duas versões da Regressão Logística ordenam os clientes igualmente bem (mesmo ROC-AUC/PR-AUC): o `class_weight='balanced'` apenas desloca o ponto de operação para um recall maior, que é o que o [ML Canvas](ml_canvas.md) pede.

#### Ajuste de hiperparâmetros (GridSearchCV com validação cruzada aninhada)

Para uma comparação justa, cada modelo passou por uma busca em grade otimizando PR-AUC, dentro de uma validação cruzada aninhada: cada um dos 5 folds externos (os mesmos da tabela anterior) roda sua própria busca em 3 folds internos, e só o fold externo mede o desempenho. Assim a estimativa não fica otimista por ter escolhido os hiperparâmetros nos mesmos dados. Resultados em `models/tuning_results.csv`.

| Modelo ajustado (CV aninhada) | Recall | F1 | ROC-AUC | PR-AUC | Hiperparâmetros finais |
|---|---|---|---|---|---|
| **Regressão Logística (balanced)** | 0,800 ± 0,028 | 0,625 ± 0,022 | 0,844 ± 0,014 | 0,653 ± 0,017 | `C=0.1`, L1 |
| Random Forest (balanced) | 0,790 ± 0,019 | 0,637 ± 0,020 | 0,847 ± 0,011 | 0,659 ± 0,023 | `max_depth=10`, `max_features=0.5`, `min_samples_leaf=20` |
| Árvore de Decisão (balanced) | 0,809 ± 0,040 | 0,627 ± 0,029 | 0,828 ± 0,013 | 0,623 ± 0,025 | `max_depth=8`, `min_samples_leaf=50` |
| MLP (sample_weight balanced) | 0,788 ± 0,029 | 0,623 ± 0,017 | 0,844 ± 0,013 | 0,653 ± 0,021 | `hidden_layer_sizes=(16,)`, `alpha=1` |

- O ajuste melhora bastante os modelos que estavam mal regularizados (a Árvore sobe de 0,66 para 0,83 de ROC-AUC) e **empata** Random Forest e MLP com a Regressão Logística: as diferenças de ROC-AUC e PR-AUC (≤ 0,006) são menores que o desvio entre folds, e as vitórias se dividem entre os folds. No conjunto de teste: ROC-AUC 0,843 (LR), 0,846 (RF), 0,844 (MLP) e 0,832 (Árvore).
- O ajuste não melhora a própria Regressão Logística (ROC-AUC 0,844 vs. 0,845), por isso o modelo em produção mantém a configuração padrão (`C=1`, L2).
- **Escolha do campeão:** com desempenho estatisticamente equivalente, vale a regra do 1 desvio-padrão: preferir o modelo mais simples. A Regressão Logística é interpretável (os coeficientes explicam o risco e permitem auditar vieses, seção 7), mais leve para servir na API e não depende de ajuste fino.

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
| `PhoneService = Yes` | −0,27 | Ter serviço de telefone associado a menor risco |
| `SeniorCitizen` | +0,26 | Idosos têm risco maior (ver seção 7) |
| `PaymentMethod = Electronic check` | +0,24 | Cheque eletrônico associado a maior risco |

`gender` tem coeficientes quase iguais para as duas categorias (−0,076 e −0,074): a diferença entre homens e mulheres é de ≈ 0,002, ou seja, praticamente sem influência. Os coeficientes são calculados em [`model_evaluation.ipynb`](../notebooks/model_evaluation.ipynb).

## 6. Limitações

1. **Precision moderada (0,52):** o modelo foi otimizado para recall, então gera muitos falsos positivos (~41% da base é sinalizada). Isso tem custo operacional e de descontos.
2. **Multicolinearidade:** `TotalCharges` ≈ `tenure × MonthlyCharges`, e o One-Hot gera colunas redundantes (ex.: `No internet service` aparece em várias variáveis). Os coeficientes devem ser interpretados com cuidado (o sinal positivo de `TotalCharges` não significa que "gastar mais no total" aumente o churn por si só).
3. **Probabilidades não calibradas:** com `class_weight='balanced'` as probabilidades ficam infladas para a classe churn (o modelo prevê ~41% de churn contra 26,5% reais). Use a saída como **ranking de risco**, não como probabilidade real.
4. **Dados estáticos e sintéticos:** o dataset da IBM é uma amostra fictícia, de uma única "foto" no tempo, sem sazonalidade, histórico de uso, reclamações ou interações com o atendimento. A performance em dados reais pode ser diferente.
5. **Desempenho desigual entre segmentos:** o modelo praticamente **não detecta churn em contratos de 2 anos** (recall 0,00 em 12 casos no teste), em clientes **sem internet** (recall 0,00 em 25 casos) e é fraco em contratos de 1 ano (0,26) e clientes com mais de 4 anos de casa (0,41). São grupos com pouco churn, mas os cancelamentos que acontecem ali passam despercebidos.
6. **Categorias novas:** valores não vistos no treino (ex.: um novo método de pagamento) não são suportados. A API rejeita essas requisições (HTTP 422); usado direto em Python, o `OneHotEncoder` (`handle_unknown='ignore'`) ignoraria a categoria em silêncio. A API também rejeita combinações inconsistentes (por exemplo, `TotalCharges` vazio com `tenure > 0`) e `tenure` acima de 120 meses. Entre 73 e 120 meses, valores acima do maior visto no treino (72), o modelo está extrapolando.
7. **Sem monitoramento de drift:** ainda não há acompanhamento de mudança na distribuição dos dados ou queda de performance em produção.

## 7. Possíveis vieses e análise de equidade

Métricas no conjunto de teste por grupo (limiar 0,5), calculadas em [`model_evaluation.ipynb`](../notebooks/model_evaluation.ipynb). *FPR = taxa de falsos positivos (clientes que não cancelariam, mas foram sinalizados).*

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
- Próximas melhorias: calibração de probabilidades (`CalibratedClassifierCV`), avaliar remoção de atributos sensíveis e ajuste do limiar conforme o custo real das ações.
- Retreinar o modelo quando houver dados reais ou drift detectado.
