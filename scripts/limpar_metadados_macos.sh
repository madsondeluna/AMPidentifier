#!/bin/bash

# Script para limpeza rápida de arquivos de metadados do macOS
# Uso: ./scripts/limpar_metadados_macos.sh [caminho]

set -e

# Se não for fornecido um caminho, usa o diretório atual
TARGET_PATH="${1:-.}"

echo "Limpando metadados do macOS em: $TARGET_PATH"
echo ""

# Contar arquivos antes
BEFORE_UNDERSCORE=$(find "$TARGET_PATH" -name "._*" -type f 2>/dev/null | wc -l | tr -d ' ')
BEFORE_DSSTORE=$(find "$TARGET_PATH" -name ".DS_Store" -type f 2>/dev/null | wc -l | tr -d ' ')

echo "Arquivos encontrados:"
echo "  ._* : $BEFORE_UNDERSCORE"
echo "  .DS_Store : $BEFORE_DSSTORE"
echo ""

if [ "$BEFORE_UNDERSCORE" -eq 0 ] && [ "$BEFORE_DSSTORE" -eq 0 ]; then
    echo "Nenhum arquivo de metadados encontrado. Tudo limpo!"
    exit 0
fi

# Remover arquivos
echo "Removendo arquivos..."
find "$TARGET_PATH" -name "._*" -type f -delete 2>/dev/null || true
find "$TARGET_PATH" -name ".DS_Store" -type f -delete 2>/dev/null || true

# Contar arquivos depois
AFTER_UNDERSCORE=$(find "$TARGET_PATH" -name "._*" -type f 2>/dev/null | wc -l | tr -d ' ')
AFTER_DSSTORE=$(find "$TARGET_PATH" -name ".DS_Store" -type f 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "Limpeza concluída!"
echo "  ._* removidos: $((BEFORE_UNDERSCORE - AFTER_UNDERSCORE))"
echo "  .DS_Store removidos: $((BEFORE_DSSTORE - AFTER_DSSTORE))"
echo ""

# Verificar se ainda há arquivos
if [ "$AFTER_UNDERSCORE" -gt 0 ] || [ "$AFTER_DSSTORE" -gt 0 ]; then
    echo "AVISO: Alguns arquivos não puderam ser removidos (permissões?)"
    echo "  ._* restantes: $AFTER_UNDERSCORE"
    echo "  .DS_Store restantes: $AFTER_DSSTORE"
fi
