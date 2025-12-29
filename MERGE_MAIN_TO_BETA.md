# Resumo: Merge Main para Beta

**Data:** 2025-12-29 01:39  
**Operação:** Merge do branch main para beta

---

## Commits Adicionados ao Beta

### 1. chore: update .gitignore (11b60c0)
- Adicionadas regras para ignorar arquivos de metadados do macOS
- Arquivos ignorados: `._*`, `.DS_Store`, `.Spotlight-V100`, `.Trashes`

### 2. feat: add maintenance scripts (361083a)
- `scripts/atualizar_comparacao_branches.sh` - Atualização automática de comparação de branches
- `scripts/configurar_hd_externo.sh` - Configuração de HD externo
- `scripts/limpar_metadados_macos.sh` - Limpeza rápida de metadados
- `scripts/README.md` - Documentação completa dos scripts

### 3. docs: add comprehensive documentation (d6a6e5d)
- `COMPARACAO_BETA_MAIN.md` - Comparação técnica detalhada entre branches
- `RESUMO_VISUAL_BRANCHES.md` - Resumo visual com atualizações automáticas
- `SOLUCAO_METADADOS_MACOS.md` - Guia completo para problema de metadados

### 4. docs: README improvements (6904cdb, 8c766ff)
- Remoção de menções redundantes ao StandardScaler
- Renomeação de seção "Benchmarking" para "Ensemble Mode Performance"

### 5. refactor: code structure (c0f9e01)
- Melhorias de legibilidade e manutenibilidade

---

## Arquivos Modificados

Total: 11 arquivos

1. `.gitignore` - Novas regras para macOS
2. `README.md` - Melhorias de documentação
3. `img/workflow.drawio` - Diagrama atualizado
4. `img/workflow.svg` - Diagrama SVG atualizado
5. `COMPARACAO_BETA_MAIN.md` - Novo
6. `RESUMO_VISUAL_BRANCHES.md` - Novo
7. `SOLUCAO_METADADOS_MACOS.md` - Novo
8. `scripts/README.md` - Novo
9. `scripts/atualizar_comparacao_branches.sh` - Novo
10. `scripts/configurar_hd_externo.sh` - Novo
11. `scripts/limpar_metadados_macos.sh` - Novo

**Linhas adicionadas:** 1018  
**Linhas removidas:** 47

---

## Status Atual dos Branches

### Beta (local e remoto)
- Commit: `d6a6e5d`
- Status: Atualizado com sucesso
- Push: Concluído para origin/beta

### Main (local)
- Commit: `d6a6e5d`
- Status: Sincronizado com beta local
- Divergente de origin/main (3 commits à frente, 6 commits atrás)

### Origin/Main (remoto)
- Commit: `b4d0192`
- Contém: Portal web e melhorias de UI (6 commits à frente)

---

## Próximos Passos

### Opção 1: Manter Main Local Como Está
- Beta está atualizado com as melhorias de documentação
- Main local pode continuar divergente de origin/main
- Útil se você não quer o portal web

### Opção 2: Sincronizar Main com Origin/Main
```bash
git checkout main
git pull origin main
```
- Trará o portal web para o main local
- Pode requerer resolução de conflitos

### Opção 3: Fazer Push Forçado do Main Local
```bash
git push origin main --force
```
- Sobrescreverá origin/main com suas mudanças
- Removerá o portal web do repositório remoto
- Use com cautela!

---

## Verificação

Para verificar o estado atual:

```bash
# Ver diferença entre beta e origin/main
git diff beta origin/main

# Ver log de commits
git log --oneline --graph --all -10

# Atualizar documento de comparação
./scripts/atualizar_comparacao_branches.sh
```

---

## Sucesso

O merge foi concluído com sucesso!  
Beta agora contém todas as melhorias de documentação e scripts de manutenção.
