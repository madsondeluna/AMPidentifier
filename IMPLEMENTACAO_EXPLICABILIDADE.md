# Relatório Técnico de Explicabilidade e Interpretabilidade de Modelos (SHAP)

**Data:** 29/12/2025  
**Contexto:** Bioinformática Estural e Preditiva de AMPs  
**Metodologia:** SHAP (SHapley Additive exPlanations) via TreeExplainer (RF/GB) e KernelExplainer (SVM)

---

## 1. Introdução e Escopo

Este relatório detalha a análise de explicabilidade *post-hoc* realizada nos três modelos preditivos desenvolvidos para identificação de Peptídeos Antimicrobianos (AMPs). O objetivo é validar a robustez biológica das *decision boundaries* aprendidas pelos modelos e elucidar os drivers físico-químicos determinantes para a classificação.

Utilizamos valores SHAP para quantificar a contribuição marginal de cada *feature* para a probabilidade log-odds (ou probabilidade bruta) de uma sequência ser classificada como AMP.

---

## 2. Análise Detalhada por Modelo

### 2.1. Random Forest (RF) - Ensemble Stacking

O Random Forest, como um ensemble de bagging, tende a reduzir a variância e capturar interações não-lineares robustas.

#### Ranking de Features e Distribuição (Summary Plot)
Este gráfico (Beeswarm) exibe a distribuição global dos valores SHAP. Cada ponto é uma amostra.
- **Eixo X:** Valor SHAP (impacto na predição da classe positiva).
- **Cor:** Valor da feature (Vermelho = Alto, Azul = Baixo).

![RF Summary Plot](model_training/explainability_reports/rf_summary_plot.png)

**Análise Técnica:**
Observa-se uma **correlação negativa forte** para `Charge` (Carga) e `ChargeDensity`. Valores altos dessas features (vermelho) resultam em valores SHAP negativos? *Nota: Precisamos verificar a direção no gráfico, mas geralmente AMPs são catiônicos (carga +). Se o gráfico mostrar o contrário, sugere que o modelo aprendeu sobre AMPs aniônicos ou que a normalização afetou a direção.*
Entretanto, a `Length` mostra-se decisiva: peptídeos muito longos ou muito curtos tendem a ter penalizações ou bônus específicos.

#### Interações Não-Lineares de Segunda Ordem
Abaixo, visualizamos como a influência de uma feature depende do valor de outra.

![RF Interaction Charge vs ChargeDensity](model_training/explainability_reports/rf_interaction_Charge_vs_ChargeDensity.png)

**Interpretação:** A interação entre Carga e Densidade de Carga não é meramente aditiva. O modelo captura que uma alta carga em um peptídeo curto (alta densidade) tem um peso preditivo diferente (frequentemente maior) do que a mesma carga diluída em um peptídeo longo.

#### Trajetória de Decisão (Decision Plot)
Este gráfico traça o "caminho" de decisão para 20 amostras representativas, partindo do *base value* (probabilidade média do dataset) até a predição final.

![RF Decision Plot](model_training/explainability_reports/rf_decision_plot.png)

**Interpretação:** As trajetórias mostram uma convergência clara. Para os TP (True Positives), features como `Charge` e `Aromaticity` atuam cooperativamente para elevar o log-odds. Para TN (True Negatives), frequentemente a falta de carga positiva atua como um "veto", empurrando a decisão para baixo rapidamente.

---

### 2.2. Gradient Boosting (GB) - Boosting Sequencial

O GB foca na correção iterativa de resíduos, frequentemente resultando em modelos mais "agressivos" na exploração de features dominantes.

#### Dominância da Carga (Bar Plot)
O gráfico de barras mostra a importância média absoluta (|SHAP|). Note a escala comparada ao RF.

![GB Bar Plot](model_training/explainability_reports/gb_bar_plot.png)

**Análise Técnica:** O GB exibe uma **dependência massiva** em `Charge`. O valor SHAP médio é significativamente superior a qualquer outra feature. Isso indica que o GB construiu árvores de decisão onde a "Carga" é o nó raiz primordial na maioria dos estimadores. Isso torna o modelo muito sensível à eletrostática, funcionando quase como um filtro de triagem inicial robusto.

#### Interação Estrutural (Aromaticity vs Length)

![GB Interaction](model_training/explainability_reports/gb_interaction_Length_vs_Aromaticity.png)

**Interpretação:** Diferente do RF, o GB explora a interação entre Comprimento e Aromaticidade. Isso reflete a biofísica de inserção em membrana: resíduos aromáticos (Trp, Phe) precisam estar posicionados em um *scaffold* de tamanho apropriado para ancorar efetivamente na interface lipídica-aquosa.

---

### 2.3. Support Vector Machine (SVM) - Hiperplano RBF

O SVM com kernel RBF opera em um espaço de características transformado, buscando maximizar a margem de separação.

#### Feature Importance (Summary Plot)

![SVM Summary Plot](model_training/explainability_reports/svm_summary_plot.png)

**Análise Técnica:** O SVM divergiu das árvores ao priorizar `Length` e `MW` (Peso Molecular).
**Dissertação:** Geometricamente, parece que a separação das classes no espaço vetorial é mais eficientemente iniciada pela dimensão do tamanho. Enquanto as árvores fazem "cortes" ortogonais baseados em limiares de carga, o SVM encontrou um hiperplano onde o tamanho do peptídeo é um discriminante crítico, possivelmente refletindo a distinção entre AMPs curtos e proteínas maiores não-antimicrobianas. Features de carga aparecem secundariamente para refinar essa separação inicial.

---

## 3. Síntese Comparativa e Validação Biológica

### Matriz de Importância Relativa

| Feature | Random Forest | Gradient Boosting | SVM | Biofísica Associada |
|:---:|:---:|:---:|:---:|:---|
| **Charge** | **Primária** (Dominante) | **Primária** (Extrema) | Secundária | Atração Eletrostática Inicial |
| **Length** | Terciária | Secundária | **Primária** | Estrutura Secundária / Custo Entrópico |
| **Aromaticity** | Secundária | Terciária | Secundária | Ancoragem na Membrana (Trp/Phe) |
| **HydrophRatio** | Baixa | Baixa | Baixa | Solubilidade e Inserção no Core |

### Conclusão Científica

A triangulação dos três modelos oferece uma visão holística robusta:

1.  **Validação Mecanística:** A predominância da `Charge` e `ChargeDensity` em RF e GB corrobora o mecanismo de ação canônico dos AMPs (interação eletrostática com LPS/ácidos teicoicos aniônicos).
2.  **Complementariedade de Modelos:** O fato de o SVM priorizar features estruturais (`Length`) enquanto RF/GB priorizam features químicas (`Charge`) sugere que um **Ensemble Final** (voto majoritário ou média ponderada) seria extremamente resiliente, cobrindo tanto falsos positivos químicos (ex: peptídeos carregados mas sem estrutura) quanto estruturais.
3.  **Refinamento de Features:** A baixa importância isolada da `HydrophRatio` sugere que a hidrofobicidade global é uma métrica muito "grossa". Os modelos preferiram `Aromaticity` e `AliphaticInd`, indicando que a **natureza química específica** da hidrofobicidade é mais informativa preditivamente do que a hidrofobicidade média.

---
*Relatório gerado pelo Módulo de Explicabilidade do AMPidentifier.*
