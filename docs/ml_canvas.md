# ML Canvas — Pipeline Preditivo de Churn

## Contexto de negócio
Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado.
A diretoria precisa identificar, com antecedência, quais clientes têm maior
risco de cancelamento.

No dataset utilizado, 26,5% dos clientes cancelaram, e eles representam 30,5%
da receita mensal: quem cancela paga, em média, 74,44 por mês, contra 61,27 de
quem fica.

## Stakeholders
- Diretoria (decide estratégia de retenção e acompanha os KPIs)
- Time de retenção/atendimento (executa ações de contato e usa o score de risco)
- Time de dados/ML (treina, serve e monitora o modelo)
- Clientes (são afetados pelas ações, por isso a análise de vieses no
  [Model Card](model_card.md#7-possíveis-vieses-e-análise-de-equidade))

## Proposta de valor
Dar ao time de retenção uma lista de clientes ordenada por risco de cancelamento,
para concentrar o orçamento de retenção em quem tem mais chance de sair.

## Tarefa de predição
Classificação binária: dado o perfil, os serviços e a conta de um cliente, estimar
a probabilidade de ele cancelar (`Churn` = Yes/No).

## Decisão que o modelo apoia
Priorizar quais clientes contatar proativamente (desconto, upgrade de
serviço, suporte) antes que cancelem, em vez de agir só depois do cliente
já ter cancelado. A decisão final continua sendo humana.

## Valor de acertar (verdadeiro positivo)
Cliente identificado corretamente como em risco → ação de retenção a tempo
→ receita recorrente preservada.

## Custo de errar
- **Falso negativo** (modelo disse que não ia cancelar, mas cancelou):
  custo alto — cliente perdido sem nenhuma ação de retenção.
- **Falso positivo** (modelo disse que ia cancelar, mas não ia):
  custo baixo/médio — contato ou desconto desnecessário, mas não perde
  o cliente.

## Dados
- **Fonte:** Telco Customer Churn (IBM, via Kaggle), com 7.043 clientes. É uma base
  fictícia, usada aqui no lugar dos dados internos da operadora.
- **Features:** dados demográficos (`gender`, `SeniorCitizen`, `Partner`,
  `Dependents`), serviços contratados (telefone, internet e adicionais) e conta
  (`tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling`, `MonthlyCharges`,
  `TotalCharges`).
- **Rótulo:** coluna `Churn`.
- **Em produção (proposta):** os dados viriam do CRM e do faturamento, e o rótulo,
  dos cancelamentos registrados.

## Como as predições são feitas
Sob demanda, pelo endpoint `POST /predict`, para um cliente ou uma lista.
**Uso proposto:** o time de retenção envia a lista de clientes ativos uma vez por mês,
antes do ciclo de campanhas, e ordena o resultado pelo score.

## Métrica técnica
Como o custo do falso negativo é maior que o do falso positivo, a métrica
principal será **recall** (idealmente acompanhado de F1-score, pra não
sacrificar demais a precisão). ROC-AUC e PR-AUC são usadas para comparar os modelos
independentemente do limiar de decisão. Acurácia não será usada como métrica
principal, pois o dataset é desbalanceado (73,5% não cancelam) e infla
artificialmente o desempenho de qualquer modelo "preguiçoso".

## Métricas de negócio
O sucesso do projeto é medido pelo resultado na operadora, não pelo desempenho do modelo.

**KPIs do projeto (só podem ser medidos com o modelo em uso):**
- **Taxa de churn mensal:** é o KPI principal. Referência no dataset: 26,5%.
- **Receita mensal perdida com churn:** referência no dataset: 30,5% da receita.
- **Retorno da campanha:** receita preservada menos o custo das ações.

Para atribuir o resultado ao modelo, é preciso manter um **grupo de controle** de
clientes sinalizados que não recebem a ação e comparar as taxas de churn. A meta de
redução deve ser definida com a diretoria.

**Indicadores que o modelo já permite estimar (conjunto de teste, 1.409 clientes):**

| Indicador | Resultado |
|---|---|
| Receita em risco detectada (limiar 0,5) | 83,3% (23.125 de 27.747 por mês) |
| Lift da lista de clientes sinalizados | 1,95 (quase o dobro de clientes em risco de uma lista aleatória) |
| Contatando só os 20% de maior risco | 52,4% dos cancelamentos e 57,5% da receita em risco |

Essas são estimativas de potencial: mostram quanta receita em risco chega ao time de
retenção, não quanta será de fato preservada. Isso depende da ação funcionar. Os cálculos,
e uma simulação de retorno com premissas hipotéticas, estão em
[`model_evaluation.ipynb`](../notebooks/model_evaluation.ipynb) e na
[seção 5.6 do Model Card](model_card.md#56-impacto-de-negócio).

## Treino e atualização
Pipeline reprodutível em `src/train_final_model.py`. **Proposta:** retreinar quando houver
dados reais da operadora, quando o monitoramento indicar queda de desempenho e, no
mínimo, a cada 6 meses.

## Monitoramento (proposta)
- **Modelo:** recall e precision depois que os cancelamentos forem conhecidos, no geral e
  por segmento (contrato, idade, forma de pagamento).
- **Dados:** mudanças na distribuição das features e na fração de clientes sinalizados
  (hoje, cerca de 41%).
- **Negócio:** os KPIs acima, comparando com o grupo de controle.
