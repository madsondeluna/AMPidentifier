# Resumo Visual: Beta vs Main

## Estrutura dos Branches

```
                    ┌─── origin/main (b4d0192) ───┐
                    │   [6 commits à frente de main]
                    │
                    ├─── main local (2ab0bd2) ─────┤
                    │   [0 commits à frente de beta]
                    │
                    └─── beta (92f887d) ───────────┘
                        [v1.0.0 - Versão Estável]
```

---

## Comparação Rápida

| Característica | Beta (v1.0.0) | Main Local | Origin/Main |
|----------------|---------------|------------|-------------|
| **Commits desde Beta** | 0 | +0 | +6 |
| **Arquivos Modificados** | - | 44 | 14 |
| **Arquivos Não Commitados** | - | 0 | - |
| **Status** | Estável | Desatualizado | Mais Recente |

---

## Status Atual do Repositório

**Última atualização:** 2025-12-29 02:54:56

### Últimos 5 Commits em Cada Branch

**Beta:**
```
  92f887d docs: add SHAP explainability reports and visualizations for RF, SVM, GB
  3b02544 fix: handle 3D SHAP values array from TreeExplainer in binary classification
  4f144ec feat: add comprehensive SHAP-based model explainability system
  cdb46cb docs: add `STATUS_ATUAL.md` detailing current repository status, configurations, and workflow.
  312992f docs: update branch comparison after final sync
```

**Main Local:**
```
  2ab0bd2 docs: add merge summary and update branch comparison
  d6a6e5d docs: add comprehensive documentation for branch comparison and macOS metadata solution
  361083a feat: add maintenance scripts for branch comparison and macOS metadata cleanup
  11b60c0 chore: update .gitignore to exclude macOS metadata files (._*, .DS_Store, .Spotlight-V100, .Trashes)
  6904cdb Remove redundant mention of StandardScaler normalization in the About section and Pre-Trained Internal Models section of README for clarity.
```

**Origin/Main:**
```
  b4d0192 Standardize navbar across all pages: update branding and GitHub link text
  0a197b8 Update navbar: change brand to 'AMPidentifier Server' and GitHub link to 'GitHub CLI'
  d23b1eb design: elegant purple/gray/white color scheme with frozen glass effect and sophisticated shadows
  60ae6cb Replace all emojis with elegant Unicode symbols for professional appearance
  30b50f6 Fix: Improve error handling for demo mode - clearer message when API not configured
```

---

## Recomendações

**ATENÇÃO:** Seu branch main local está 6 commit(s) atrás de origin/main.

Para sincronizar:
```bash
git pull origin main
```

