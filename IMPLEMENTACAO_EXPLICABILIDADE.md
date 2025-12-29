# Relatório de Explicabilidade de Modelos de Predição de AMPs

**Data da Análise:** 29/12/2025  
**Modelos Analisados:** Random Forest (RF), Support Vector Machine (SVM), Gradient Boosting (GB)  
**Método:** SHAP (SHapley Additive exPlanations)

---

## 1. Introdução Visual e Resultados Principais

Este documento apresenta a "caixa aberta" dos nossos modelos de inteligência artificial. Utilizamos o método SHAP para desvendar exatamente quais propriedades físico-químicas cada modelo utiliza para classificar um peptídeo como Antimicrobiano (AMP) ou não.

### Galeria de Explicabilidade

Abaixo estão os gráficos chave gerados pela análise. Todos os arquivos encontram-se em `model_training/explainability_reports/`.

#### Análise Global (Random Forest)
O **Summary Plot** é a visão geral mais importante. Ele mostra quais features são mais impactantes e como elas afetam a decisão.
*(Ver `rf_summary_plot.png`)*

#### Caminhos de Decisão
O **Decision Plot** ilustra como o modelo chega a uma conclusão para diferentes amostras, somando as contribuições de cada feature.
*(Ver `rf_decision_plot.png` e `svm_decision_plot.png`)*

#### Interações Complexas
Detectamos que o modelo não olha para features isoladamente. O gráfico abaixo mostra como a **Carga (Charge)** e a **Densidade de Carga** interagem sinergicamente.
*(Ver `rf_interaction_Charge_vs_ChargeDensity.png`)*

---

## 2. Análise Comparativa Detalhada

### Tabela de Consenso de Features (Top 5)

A tabela abaixo compara o ranking de importância das features entre os três modelos.

| Rank | Random Forest (RF) | Gradient Boosting (GB) | SVM | Consenso Científico |
|:---:|:---|:---|:---|:---|
| **#1** | **Charge** (Carga) | **Charge** (Carga) | **Length** (Comprimento) | **Alta Concordância** |
| **#2** | **ChargeDensity** | **ChargeDensity** | **ChargeDensity** | Modelos concordam na importância da Carga |
| **#3** | Aromaticity | Length | MW (Peso Molecular) | Divergência em features estruturais |
| **#4** | Length | MW | Charge | RF/GB priorizam química, SVM prioriza geometria |
| **#5** | pI | Aromaticity | pI | |

### Análise Crítica dos Modelos

#### Modelo 1: Random Forest (O "Químico")
*   **Foco:** Puramente eletrostático. Carga e Densidade de Carga dominam.
*   **Insight:** O RF aprendeu corretamente que a atração magnética inicial (catônica vs aniônica) é o filtro mais forte para AMPs.
*   **Descoberta de Interação:** O RF detectou uma interação interessante entre `Length` e `Aromaticity`, sugerindo que o tamanho do peptídeo modula a importância de resíduos aromáticos para a estabilidade na membrana.

#### Modelo 2: Gradient Boosting (O "Especialista")
*   **Foco:** Extremamente focado em **Charge** (o valor SHAP é quase 10x maior que outras features).
*   **Comportamento:** É o modelo mais "opinativo". Se não tiver carga positiva, ele descarta a possibilidade de ser AMP muito rapidamente.
*   **Vantagem:** Reduz falsos positivos em peptídeos neutros.

#### Modelo 3: SVM (O "Geômetra")
*   **Foco Principal:** `Length` e `MW`.
*   **Diferença:** Ao contrário das árvores, o SVM prioriza o tamanho do peptídeo como fator discriminante primário.
*   **Interpretação:** Isso sugere que o SVM está separando as classes baseado em um hiperplano onde o comprimento ajuda a "fatiar" o espaço de dados melhor do que a carga sozinha. Isso complementa muito bem os outros modelos em um ensemble.

---

## 3. Discussão Científica

### O "Dogma Central" dos Nossos Modelos
Existe um consenso robusto: **A Eletrostática Domina**.
Todos os modelos, de formas diferentes, concordam que a carga líquida positiva é um preditor fundamental. Isso valida biologicamente os modelos, pois o mecanismo de ação primário dos AMPs é a interação com membranas bacterianas carregadas negativamente.

### O Papel da Hidrofobicidade
Curiosamente, `HydrophRatio` aparece consistentemente nas últimas posições (Rank #10).
*   **Por que?** Provavelmente porque a hidrofobicidade é capturada de forma mais específica por `Aromaticity` e `AliphaticInd`. O modelo prefere tipos específicos de resíduos hidrofóbicos (aromáticos como Triptofano) do que uma métrica genérica de hidrofobicidade.

### Confiabilidade e Transparência
As análises `plot_decision_plot` mostram trajetórias claras e distintas para AMPs e Não-AMPs, indicando que os modelos não estão "chutando" ou se baseando em artefatos, mas seguindo um caminho lógico de decisão baseado em propriedades físico-químicas reais.

---

## 4. Conclusão

A implementação do SHAP transformou modelos "caixa preta" em ferramentas transparentes. Podemos afirmar com segurança que:
1.  Os modelos predizem AMPs baseados em **biologia real** (Carga, Tamanho, Estrutura).
2.  Eles são complementares: RF/GB capturam a química fina, enquanto SVM captura a geometria global.
3.  Eles são auditáveis: qualquer predição individual pode ser explicada através dos *Waterfall Plots*.

---
*Relatório gerado automaticamente a partir da análise SHAP do AMPidentifier.*
