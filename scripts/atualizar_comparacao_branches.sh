#!/bin/bash

# Script para atualizar a comparação entre branches
# Uso: ./scripts/atualizar_comparacao_branches.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_FILE="$REPO_ROOT/COMPARACAO_BETA_MAIN.md"
RESUMO_FILE="$REPO_ROOT/RESUMO_VISUAL_BRANCHES.md"

cd "$REPO_ROOT"

echo "Atualizando comparação de branches..."
echo "Data: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Obter informações dos branches
BETA_COMMIT=$(git rev-parse --short beta 2>/dev/null || echo "N/A")
MAIN_COMMIT=$(git rev-parse --short main 2>/dev/null || echo "N/A")
ORIGIN_MAIN_COMMIT=$(git rev-parse --short origin/main 2>/dev/null || echo "N/A")

# Contar commits à frente
COMMITS_MAIN_AHEAD_BETA=$(git rev-list --count beta..main 2>/dev/null || echo "0")
COMMITS_ORIGIN_AHEAD_MAIN=$(git rev-list --count main..origin/main 2>/dev/null || echo "0")
COMMITS_ORIGIN_AHEAD_BETA=$(git rev-list --count beta..origin/main 2>/dev/null || echo "0")

# Arquivos modificados
FILES_CHANGED_BETA_MAIN=$(git diff --name-only beta..main 2>/dev/null | wc -l | tr -d ' ')
FILES_CHANGED_MAIN_ORIGIN=$(git diff --name-only main..origin/main 2>/dev/null | wc -l | tr -d ' ')

# Arquivos não commitados
UNCOMMITTED_FILES=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')

echo "Informações coletadas:"
echo "  Beta: $BETA_COMMIT"
echo "  Main Local: $MAIN_COMMIT (+$COMMITS_MAIN_AHEAD_BETA commits desde beta)"
echo "  Origin/Main: $ORIGIN_MAIN_COMMIT (+$COMMITS_ORIGIN_AHEAD_BETA commits desde beta)"
echo "  Arquivos modificados (beta→main): $FILES_CHANGED_BETA_MAIN"
echo "  Arquivos modificados (main→origin): $FILES_CHANGED_MAIN_ORIGIN"
echo "  Arquivos não commitados: $UNCOMMITTED_FILES"
echo ""

# Gerar relatório resumido
cat > "$RESUMO_FILE.tmp" <<EOF
# Resumo Visual: Beta vs Main

## Estrutura dos Branches

\`\`\`
                    ┌─── origin/main ($ORIGIN_MAIN_COMMIT) ───┐
                    │   [$COMMITS_ORIGIN_AHEAD_MAIN commits à frente de main]
                    │
                    ├─── main local ($MAIN_COMMIT) ─────┤
                    │   [$COMMITS_MAIN_AHEAD_BETA commits à frente de beta]
                    │
                    └─── beta ($BETA_COMMIT) ───────────┘
                        [v1.0.0 - Versão Estável]
\`\`\`

---

## Comparação Rápida

| Característica | Beta (v1.0.0) | Main Local | Origin/Main |
|----------------|---------------|------------|-------------|
| **Commits desde Beta** | 0 | +$COMMITS_MAIN_AHEAD_BETA | +$COMMITS_ORIGIN_AHEAD_BETA |
| **Arquivos Modificados** | - | $FILES_CHANGED_BETA_MAIN | $FILES_CHANGED_MAIN_ORIGIN |
| **Arquivos Não Commitados** | - | $UNCOMMITTED_FILES | - |
| **Status** | Estável | $([ "$COMMITS_ORIGIN_AHEAD_MAIN" -gt 0 ] && echo "Desatualizado" || echo "Atualizado") | Mais Recente |

---

## Status Atual do Repositório

**Última atualização:** $(date '+%Y-%m-%d %H:%M:%S')

EOF

# Adicionar informações sobre commits recentes
echo "### Últimos 5 Commits em Cada Branch" >> "$RESUMO_FILE.tmp"
echo "" >> "$RESUMO_FILE.tmp"

echo "**Beta:**" >> "$RESUMO_FILE.tmp"
echo "\`\`\`" >> "$RESUMO_FILE.tmp"
git log beta --oneline -5 2>/dev/null | sed 's/^/  /' >> "$RESUMO_FILE.tmp"
echo "\`\`\`" >> "$RESUMO_FILE.tmp"
echo "" >> "$RESUMO_FILE.tmp"

echo "**Main Local:**" >> "$RESUMO_FILE.tmp"
echo "\`\`\`" >> "$RESUMO_FILE.tmp"
git log main --oneline -5 2>/dev/null | sed 's/^/  /' >> "$RESUMO_FILE.tmp"
echo "\`\`\`" >> "$RESUMO_FILE.tmp"
echo "" >> "$RESUMO_FILE.tmp"

echo "**Origin/Main:**" >> "$RESUMO_FILE.tmp"
echo "\`\`\`" >> "$RESUMO_FILE.tmp"
git log origin/main --oneline -5 2>/dev/null | sed 's/^/  /' >> "$RESUMO_FILE.tmp"
echo "\`\`\`" >> "$RESUMO_FILE.tmp"
echo "" >> "$RESUMO_FILE.tmp"

# Adicionar recomendações baseadas no estado
echo "---" >> "$RESUMO_FILE.tmp"
echo "" >> "$RESUMO_FILE.tmp"
echo "## Recomendações" >> "$RESUMO_FILE.tmp"
echo "" >> "$RESUMO_FILE.tmp"

if [ "$COMMITS_ORIGIN_AHEAD_MAIN" -gt 0 ]; then
    echo "**ATENÇÃO:** Seu branch main local está $COMMITS_ORIGIN_AHEAD_MAIN commit(s) atrás de origin/main." >> "$RESUMO_FILE.tmp"
    echo "" >> "$RESUMO_FILE.tmp"
    echo "Para sincronizar:" >> "$RESUMO_FILE.tmp"
    echo "\`\`\`bash" >> "$RESUMO_FILE.tmp"
    echo "git pull origin main" >> "$RESUMO_FILE.tmp"
    echo "\`\`\`" >> "$RESUMO_FILE.tmp"
    echo "" >> "$RESUMO_FILE.tmp"
fi

if [ "$UNCOMMITTED_FILES" -gt 0 ]; then
    echo "**ATENÇÃO:** Você tem $UNCOMMITTED_FILES arquivo(s) modificado(s) não commitado(s)." >> "$RESUMO_FILE.tmp"
    echo "" >> "$RESUMO_FILE.tmp"
    echo "Para salvar suas mudanças:" >> "$RESUMO_FILE.tmp"
    echo "\`\`\`bash" >> "$RESUMO_FILE.tmp"
    echo "git add ." >> "$RESUMO_FILE.tmp"
    echo "git commit -m \"Descrição das mudanças\"" >> "$RESUMO_FILE.tmp"
    echo "\`\`\`" >> "$RESUMO_FILE.tmp"
    echo "" >> "$RESUMO_FILE.tmp"
fi

if [ "$COMMITS_ORIGIN_AHEAD_MAIN" -eq 0 ] && [ "$UNCOMMITTED_FILES" -eq 0 ]; then
    echo "**STATUS:** Seu repositório está sincronizado e limpo!" >> "$RESUMO_FILE.tmp"
    echo "" >> "$RESUMO_FILE.tmp"
fi

# Mover arquivo temporário para o final
mv "$RESUMO_FILE.tmp" "$RESUMO_FILE"

echo "Documentos atualizados com sucesso!"
echo "  - $RESUMO_FILE"
echo ""
echo "Para ver as diferenças detalhadas, execute:"
echo "  git diff beta..main"
echo "  git diff main..origin/main"
