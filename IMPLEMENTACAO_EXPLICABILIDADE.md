# Implementação de Explicabilidade com SHAP

**Data:** 2025-12-29  
**Branch:** beta  
**Status:** Concluído ✅

---

## Resumo Executivo

Foi implementado um sistema completo de explicabilidade para os três modelos de predição de AMPs usando SHAP. A análise foi executada com sucesso e gerou relatórios detalhados que comprovam a transparência dos modelos.

### Principais Resultados da Análise

- **Random Forest:** `Charge` (Carga) e `ChargeDensity` são as features mais decisivas.
- **SVM:** `Length` (Comprimento) e `ChargeDensity` dominam a decisão.
- **Gradient Boosting:** Altamente focado em `Charge`, seguido por `ChargeDensity`.

Existe um consenso claro entre os modelos de que propriedades relacionadas à **Carga** (Charge, ChargeDensity) e **Estrutura** (Length, MW) são fundamentais para identificar Peptídeos Antimicrobianos, o que é biologicamente coerente.

Todas as visualizações e tabelas foram geradas e salvas em `model_training/explainability_reports/`.

---

## Arquivos Criados

### 1. Módulo Principal
**`model_training/explainability.py`** (500+ linhas)

Módulo Python completo com funções para:
- Carregar modelos e dados
- Criar explainers SHAP apropriados para cada tipo de modelo
- Calcular valores SHAP
- Gerar múltiplas visualizações
- Criar tabelas de importância de features
- Gerar relatórios comparativos

### 2. Script de Execução
**`scripts/run_explainability_analysis.sh`**

Script bash para:
- Verificar dependências
- Instalar pacotes necessários
- Executar análise completa
- Reportar progresso e resultados

### 3. Documentação
**`model_training/EXPLAINABILITY_README.md`**

README completo com:
- Explicação do SHAP
- Instruções de uso
- Interpretação de resultados
- Troubleshooting
- Melhores práticas
- Exemplos de extensão

### 4. Dependências
**`requirements.txt`** (atualizado)

Adicionadas bibliotecas:
- `shap>=0.42.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.12.0`

---

## Funcionalidades Implementadas

### Análise por Modelo

Para cada um dos 3 modelos (RF, SVM, GB):

#### 1. Summary Plot (Beeswarm)
- Mostra distribuição de valores SHAP para cada feature
- Features ranqueadas por importância
- Cor indica valor da feature (vermelho = alto, azul = baixo)
- Arquivo: `{model}_summary_plot.png`

#### 2. Bar Plot
- Importância global das features (média dos valores SHAP absolutos)
- Ranking simples e direto
- Arquivo: `{model}_bar_plot.png`

#### 3. Waterfall Plots (3 exemplos por modelo)
- Explicação de predições individuais
- Mostra como cada feature contribui para uma predição específica
- Demonstra processo de tomada de decisão
- Arquivos: `{model}_waterfall_sample_1.png`, `_2.png`, `_3.png`

#### 4. Dependence Plots (top 5 features por modelo)
- Relação entre valores de features e valores SHAP
- Revela relações não-lineares
- Mostra interações entre features
- Arquivos: `{model}_dependence_{feature}.png`

#### 5. Tabela de Importância
- Ranking completo de features
- Valores SHAP médios (absolutos e direcionais)
- Arquivo: `{model}_feature_importance.csv`

### Análise Comparativa

#### 1. Gráfico de Comparação
- Top 15 features mais importantes
- Comparação lado a lado dos 3 modelos
- Identifica features consensuais
- Arquivo: `models_comparison.png`

#### 2. Tabela de Comparação
- Valores de importância para todos os modelos
- Formato CSV para análise adicional
- Arquivo: `models_comparison.csv`

#### 3. Relatório Markdown Completo
- Sumário executivo
- Explicação do SHAP
- Análise detalhada por modelo
- Top 10 features de cada modelo
- Guia de interpretação
- Conclusões
- Arquivo: `EXPLAINABILITY_REPORT.md`

---

## Tipos de SHAP Explainers Utilizados

### TreeExplainer (RF e GB)
- Rápido e exato para modelos baseados em árvores
- Usa estrutura da árvore para computação eficiente
- Fornece valores Shapley exatos

### KernelExplainer (SVM)
- Abordagem model-agnostic
- Usa amostragem para aproximação
- Mais lento mas funciona para qualquer modelo
- Usa 100 amostras de background para eficiência

---

## Saídas Geradas

### Estrutura de Diretórios

```
model_training/
├── explainability.py                    # Módulo principal
├── EXPLAINABILITY_README.md             # Documentação
└── explainability_reports/              # Diretório de saída
    ├── EXPLAINABILITY_REPORT.md         # Relatório completo
    ├── rf_summary_plot.png              # RF: Summary
    ├── rf_bar_plot.png                  # RF: Bar
    ├── rf_waterfall_sample_1.png        # RF: Waterfall 1
    ├── rf_waterfall_sample_2.png        # RF: Waterfall 2
    ├── rf_waterfall_sample_3.png        # RF: Waterfall 3
    ├── rf_dependence_{feature}.png      # RF: Dependence (5 files)
    ├── rf_feature_importance.csv        # RF: Tabela
    ├── svm_summary_plot.png             # SVM: Summary
    ├── svm_bar_plot.png                 # SVM: Bar
    ├── svm_waterfall_sample_1.png       # SVM: Waterfall 1
    ├── svm_waterfall_sample_2.png       # SVM: Waterfall 2
    ├── svm_waterfall_sample_3.png       # SVM: Waterfall 3
    ├── svm_dependence_{feature}.png     # SVM: Dependence (5 files)
    ├── svm_feature_importance.csv       # SVM: Tabela
    ├── gb_summary_plot.png              # GB: Summary
    ├── gb_bar_plot.png                  # GB: Bar
    ├── gb_waterfall_sample_1.png        # GB: Waterfall 1
    ├── gb_waterfall_sample_2.png        # GB: Waterfall 2
    ├── gb_waterfall_sample_3.png        # GB: Waterfall 3
    ├── gb_dependence_{feature}.png      # GB: Dependence (5 files)
    ├── gb_feature_importance.csv        # GB: Tabela
    ├── models_comparison.png            # Comparação
    └── models_comparison.csv            # Tabela comparação
```

**Total de arquivos:** ~35 arquivos
- 3 summary plots
- 3 bar plots
- 9 waterfall plots
- 15 dependence plots
- 3 tabelas CSV de importância
- 1 gráfico de comparação
- 1 tabela de comparação
- 1 relatório Markdown

---

## Como Usar

### Instalação de Dependências

```bash
pip install -r requirements.txt
```

### Execução

#### Opção 1: Script (Recomendado)
```bash
./scripts/run_explainability_analysis.sh
```

#### Opção 2: Python Direto
```bash
python3 -m model_training.explainability
```

### Visualização dos Resultados

```bash
# Ver relatório completo
cat model_training/explainability_reports/EXPLAINABILITY_REPORT.md

# Listar todos os arquivos gerados
ls -lh model_training/explainability_reports/

# Abrir imagens (macOS)
open model_training/explainability_reports/*.png
```

---

## Tempo de Execução Estimado

- **Random Forest:** ~1-2 minutos
- **Gradient Boosting:** ~1-2 minutos  
- **SVM:** ~5-10 minutos (KernelExplainer é mais lento)

**Total:** ~10-15 minutos

---

## Interpretação dos Resultados

### Valores SHAP

- **Valor SHAP positivo:** Feature empurra predição para classe positiva (AMP)
- **Valor SHAP negativo:** Feature empurra predição para classe negativa (não-AMP)
- **Magnitude:** Valor absoluto maior = influência mais forte

### Cores nos Summary Plots

- **Vermelho:** Valor alto da feature
- **Azul:** Valor baixo da feature
- **Roxo:** Valor médio da feature

### Exemplo de Interpretação

Se uma feature tem:
- Alto valor SHAP quando vermelha (valor alto) → Valores altos predizem AMP
- Baixo valor SHAP quando azul (valor baixo) → Valores baixos predizem não-AMP
- Isso indica correlação positiva com predição de AMP

---

## Casos de Uso

### 1. Validação de Modelos
- Verificar se modelos usam features biologicamente relevantes
- Confirmar que features de carga são importantes (esperado para AMPs)
- Validar que hidrofobicidade é significativa (interação com membrana)

### 2. Engenharia de Features
- Identificar features mais importantes
- Focar em features relevantes para melhorias
- Remover ou combinar features menos importantes

### 3. Confiança e Transparência
- Demonstrar interpretabilidade para stakeholders
- Construir confiança nas predições
- Identificar potenciais vieses

### 4. Insights Científicos
- Descobrir quais propriedades físico-químicas definem AMPs
- Comparar como diferentes modelos priorizam features
- Identificar features inesperadamente importantes

---

## Próximos Passos

### Integração com Pipeline Principal

```bash
# 1. Treinar modelos
python3 -m model_training.train

# 2. Avaliar modelos
python3 -m model_training.evaluate

# 3. Gerar relatórios de explicabilidade
./scripts/run_explainability_analysis.sh

# 4. Revisar resultados
cat model_training/explainability_reports/EXPLAINABILITY_REPORT.md
```

### Possíveis Extensões

1. **Análise de Subgrupos**
   - Explicar predições de alta vs baixa confiança
   - Analisar AMPs vs não-AMPs separadamente

2. **Visualizações Interativas**
   - Usar SHAP force plots interativos
   - Criar dashboard com Streamlit

3. **Análise Temporal**
   - Comparar explicabilidade entre versões de modelos
   - Rastrear mudanças em importância de features

4. **Integração com Predições**
   - Adicionar explicações SHAP ao output de predições
   - Gerar relatórios individuais para cada sequência

---

## Benefícios

### Para Pesquisadores
- Entendimento profundo do comportamento dos modelos
- Validação científica das predições
- Identificação de padrões biológicos

### Para Usuários
- Confiança nas predições
- Transparência no processo de decisão
- Capacidade de questionar e validar resultados

### Para o Projeto
- Demonstração de que modelos não são caixas pretas
- Documentação completa de explicabilidade
- Base para publicações científicas

---

## Conclusão

A implementação do sistema de explicabilidade com SHAP fornece:

1. **Transparência completa** dos modelos de predição de AMPs
2. **Múltiplas visualizações** para diferentes perspectivas
3. **Documentação abrangente** para uso e interpretação
4. **Comparação entre modelos** para identificar consenso
5. **Base científica** para validação de predições

Os modelos agora são **completamente interpretáveis** e **não são caixas pretas**, com evidências visuais e quantitativas de como cada feature contribui para as predições.

---

## Referências

- Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions" (NIPS)
- Lundberg et al. (2020) "From local explanations to global understanding with explainable AI"
- SHAP Documentation: https://shap.readthedocs.io/
- SHAP GitHub: https://github.com/slundberg/shap
