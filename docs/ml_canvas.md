# ML Canvas — Pipeline Preditivo de Churn

## Contexto de negócio
Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado.
A diretoria precisa identificar, com antecedência, quais clientes têm maior
risco de cancelamento.

## Stakeholders
- Diretoria (decide estratégia de retenção)
- Time de retenção/atendimento (executa ações de contato)

## Decisão que o modelo apoia
Priorizar quais clientes contatar proativamente (desconto, upgrade de
serviço, suporte) antes que cancelem, em vez de agir só depois do cliente
já ter cancelado.

## Valor de acertar (verdadeiro positivo)
Cliente identificado corretamente como em risco → ação de retenção a tempo
→ receita recorrente preservada.

## Custo de errar
- **Falso negativo** (modelo disse que não ia cancelar, mas cancelou):
  custo alto — cliente perdido sem nenhuma ação de retenção.
- **Falso positivo** (modelo disse que ia cancelar, mas não ia):
  custo baixo/médio — contato ou desconto desnecessário, mas não perde
  o cliente.

## Métrica técnica
Como o custo do falso negativo é maior que o do falso positivo, a métrica
principal será **recall** (idealmente acompanhado de F1-score, pra não
sacrificar demais a precisão). Acurácia não será usada como métrica
principal, pois o dataset é desbalanceado (73,5% não cancelam) e infla
artificialmente o desempenho de qualquer modelo "preguiçoso".

## Métrica de negócio
Percentual de clientes que de fato cancelaram e que foram corretamente
identificados pelo modelo como "em risco" (equivalente ao recall, mas
expresso em termos de negócio).