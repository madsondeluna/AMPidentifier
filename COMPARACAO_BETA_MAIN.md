# Comparação entre Branch Beta e Main (Local)

**Data da análise:** 2025-12-29

## Resumo Executivo

Este documento compara as diferenças entre:
- **Branch Beta** (versão `0d68c57`, tag `AMPidentifierv1.0.0`)
- **Branch Main Local** (versão `6904cdb`)
- **Branch Main Remoto** (origin/main, versão `b4d0192`)

---

## Estrutura dos Branches

```
Beta (0d68c57) ─────┬──── Main Local (6904cdb)
                    │
                    └──── Origin/Main (b4d0192)
                              │
                              └── Commits adicionais:
                                  - Portal Web
                                  - Melhorias de UI
                                  - Atualizações de navbar
```

---

## Diferenças: Beta → Main Local

### Commits Adicionados no Main Local (3 commits)

1. **`6904cdb`** - Remove redundant mention of StandardScaler normalization in the About section and Pre-Trained Internal Models section of README for clarity.

2. **`8c766ff`** - Update README: Rename "Benchmarking (Using the Ensemble Mode) - Real Data" to "Ensemble Mode Performance" for clarity

3. **`c0f9e01`** - Refactor code structure for improved readability and maintainability

### Arquivos Modificados

Apenas **3 arquivos** foram alterados entre Beta e Main Local:

1. **`README.md`** (12 linhas alteradas)
2. **`img/workflow.drawio`** (91 linhas alteradas)
3. **`img/workflow.svg`** (2 linhas alteradas)

### Mudanças Específicas no README.md

#### 1. Remoção de Menções ao StandardScaler

**Beta:**
```markdown
The **AMPidentifier** is a Python tool for predicting and analyzing Antimicrobial Peptides (AMPs) 
from amino-acid sequences. It leverages a set of pre-trained Machine Learning models with 
**StandardScaler normalization** and offers flexible prediction modes...
```

**Main Local:**
```markdown
The **AMPidentifier** is a Python tool for predicting and analyzing Antimicrobial Peptides (AMPs) 
from amino-acid sequences. It leverages a set of pre-trained Machine Learning models and offers 
flexible prediction modes...
```

**Impacto:** Simplificação da descrição, removendo detalhes técnicos de normalização que podem confundir usuários.

#### 2. Renomeação de Seção

**Beta:**
```markdown
## Benchmarking (Using the Ensemble Mode) - Real Data
**Performance with Normalized Models (StandardScaler)**
```

**Main Local:**
```markdown
## Ensemble Mode Performance
```

**Impacto:** Título mais conciso e direto.

#### 3. Remoção de Redundância na Seção de Modelos

**Beta:**
```markdown
Three models are distributed and evaluated on the same dataset for fair comparison. 
All models are trained with **StandardScaler normalization** for optimal performance.
```

**Main Local:**
```markdown
Three models are distributed and evaluated on the same dataset for fair comparison.
```

**Impacto:** Evita repetição de informações técnicas já mencionadas em outros lugares.

#### 4. Atualização do Índice (Table of Contents)

**Beta:**
```markdown
- [Benchmarking (Using the Ensemble Mode) - Real Data](#benchmarking-using-the-ensemble-mode---real-data)
```

**Main Local:**
```markdown
- [Ensemble Mode Performance](#ensemble-mode-performance)
```

---

## Diferenças: Main Local → Origin/Main (Remoto)

O **origin/main** está **6 commits à frente** do main local. Estes commits adicionam funcionalidades importantes:

### Commits Adicionais no Origin/Main (6 commits)

1. **`bc7e328`** - Add AMPidentifier web portal with minimalist design
   - **Novo recurso:** Portal web completo para o AMPidentifier

2. **`30b50f6`** - Fix: Improve error handling for demo mode - clearer message when API not configured
   - **Melhoria:** Melhor tratamento de erros no modo demo

3. **`60ae6cb`** - Replace all emojis with elegant Unicode symbols for professional appearance
   - **UI/UX:** Substituição de emojis por símbolos Unicode elegantes

4. **`d23b1eb`** - design: elegant purple/gray/white color scheme with frozen glass effect and sophisticated shadows
   - **Design:** Esquema de cores elegante com efeito de vidro congelado

5. **`0a197b8`** - Update navbar: change brand to 'AMPidentifier Server' and GitHub link to 'GitHub CLI'
   - **UI:** Atualização da navbar

6. **`b4d0192`** - Standardize navbar across all pages: update branding and GitHub link text
   - **Padronização:** Navbar consistente em todas as páginas

### Funcionalidades Presentes Apenas no Origin/Main

- **Portal Web Completo** (HTML/CSS/JavaScript)
- **Interface de Usuário Moderna** com glassmorphism
- **Modo Demo** com tratamento de erros aprimorado
- **Design Profissional** com esquema de cores roxo/cinza/branco
- **Navegação Padronizada** em todas as páginas

---

## Mudanças Não Commitadas no Main Local

Há **65 arquivos modificados** no diretório de trabalho que **não foram commitados**:

### Categorias de Mudanças Não Commitadas

1. **Código Python** (módulos principais)
   - `amp_identifier/*.py`
   - `model_training/*.py`
   - `main.py`

2. **Arquivos de Cache** (`.pyc`)
   - Múltiplos arquivos `__pycache__/*.pyc`

3. **Dados de Benchmarking**
   - `benchmarking/base/*.fasta`

4. **Resultados de Testes**
   - `data-for-tests/*/prediction_comparison_report.csv`
   - `data-for-tests/*/physicochemical_features.csv`

5. **Modelos Salvos**
   - `model_training/saved_model/*.pkl`
   - `model_training/saved_model/*.csv`
   - `model_training/saved_model/*.txt`

6. **Documentação**
   - `normalization-info/*.md`

7. **Imagens**
   - `img/*.png`
   - `img/workflow.*`

8. **Configuração**
   - `.gitignore`
   - `requirements.txt`

---

## Recomendações

### 1. Para Sincronizar com Origin/Main

Se você deseja ter o portal web e as melhorias de UI:

```bash
# Salvar mudanças locais (se necessário)
git stash

# Atualizar o branch main local
git pull origin main

# Recuperar mudanças locais (se necessário)
git stash pop
```

### 2. Para Manter Apenas as Mudanças do Main Local

Se você prefere manter apenas as alterações de documentação:

```bash
# Commitar as mudanças não salvas primeiro
git add README.md img/workflow.*
git commit -m "Update documentation and workflow diagrams"

# Decidir se quer fazer merge ou não com origin/main
```

### 3. Para Limpar Arquivos de Cache

Os arquivos `.pyc` e `__pycache__` não deveriam estar marcados como modificados. Recomendo:

```bash
# Adicionar ao .gitignore se ainda não estiver
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore

# Remover do tracking do git
git rm -r --cached amp_identifier/__pycache__
git rm -r --cached model_training/__pycache__
```

---

## Tabela Comparativa

| Aspecto | Beta | Main Local | Origin/Main |
|---------|------|------------|-------------|
| **Versão** | v1.0.0 (tag) | Ahead +3 commits | Ahead +9 commits |
| **Portal Web** | Não | Não | Sim |
| **Documentação Simplificada** | Não | Sim | Sim |
| **UI Moderna** | Não | Não | Sim |
| **Workflow Atualizado** | Não | Sim | Sim |
| **Modo Demo** | Não | Não | Sim |

---

## Observações Importantes

1. **Problema com Arquivos macOS:** Há arquivos `._*` no repositório Git causando erros:
   ```
   error: non-monotonic index .git/objects/pack/._pack-*.idx
   ```
   Estes são arquivos de metadados do macOS que não deveriam estar no repositório.

2. **Divergência de Branches:** O main local e o origin/main divergiram. O main local tem commits de documentação, enquanto o origin/main tem o portal web.

3. **Mudanças Não Commitadas:** Há muitas mudanças não commitadas que podem ser perdidas se não forem salvas adequadamente.

---

## Conclusão

- **Beta** é a versão estável marcada como v1.0.0
- **Main Local** tem melhorias de documentação (+3 commits)
- **Origin/Main** tem portal web completo e melhorias de UI (+6 commits adicionais)

**Próximos Passos Sugeridos:**
1. Decidir qual linha de desenvolvimento seguir (documentação vs portal web)
2. Fazer merge ou rebase conforme necessário
3. Limpar arquivos de cache e metadados do macOS
4. Commitar ou descartar as mudanças não salvas
