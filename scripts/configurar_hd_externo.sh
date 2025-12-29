#!/bin/bash

# Script para configurar HD externo e prevenir criação de arquivos ._* do macOS
# Uso: ./scripts/configurar_hd_externo.sh [caminho_do_volume]

set -e

# Se não for fornecido um caminho, usa o volume atual
VOLUME_PATH="${1:-/Volumes/promethion}"

echo "Configurando volume: $VOLUME_PATH"
echo ""

# Verificar se o volume existe
if [ ! -d "$VOLUME_PATH" ]; then
    echo "ERRO: Volume $VOLUME_PATH não encontrado!"
    exit 1
fi

# 1. Remover arquivos ._ existentes
echo "1. Removendo arquivos ._* existentes..."
find "$VOLUME_PATH" -name "._*" -type f -delete 2>/dev/null || true
REMOVED_COUNT=$(find "$VOLUME_PATH" -name "._*" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "   Arquivos ._* removidos"

# 2. Remover arquivos .DS_Store
echo "2. Removendo arquivos .DS_Store..."
find "$VOLUME_PATH" -name ".DS_Store" -type f -delete 2>/dev/null || true
echo "   Arquivos .DS_Store removidos"

# 3. Remover diretórios .Spotlight-V100 e .Trashes
echo "3. Removendo diretórios de metadados do macOS..."
rm -rf "$VOLUME_PATH/.Spotlight-V100" 2>/dev/null || true
rm -rf "$VOLUME_PATH/.Trashes" 2>/dev/null || true
rm -rf "$VOLUME_PATH/.fseventsd" 2>/dev/null || true
echo "   Diretórios de metadados removidos"

# 4. Criar arquivo .metadata_never_index para prevenir indexação do Spotlight
echo "4. Desabilitando indexação do Spotlight..."
touch "$VOLUME_PATH/.metadata_never_index"
echo "   Spotlight desabilitado"

# 5. Configurar atributos estendidos para prevenir criação de ._*
echo "5. Configurando atributos do volume..."

# Desabilitar criação de arquivos .DS_Store em volumes de rede
# (funciona para alguns tipos de volumes externos)
defaults write com.apple.desktopservices DSDontWriteNetworkStores -bool true
defaults write com.apple.desktopservices DSDontWriteUSBStores -bool true

echo "   Atributos configurados"

# 6. Criar arquivo .noindex na raiz para evitar indexação
echo "6. Criando marcador .noindex..."
touch "$VOLUME_PATH/.noindex" 2>/dev/null || true
echo "   Marcador criado"

echo ""
echo "Configuração concluída!"
echo ""
echo "Configurações aplicadas:"
echo "  - Arquivos ._* removidos"
echo "  - Arquivos .DS_Store removidos"
echo "  - Spotlight desabilitado"
echo "  - Indexação desabilitada"
echo "  - Criação de metadados em USB/rede desabilitada"
echo ""
echo "IMPORTANTE:"
echo "  - Essas configurações são aplicadas no macOS atual"
echo "  - Ao conectar em outro Mac, execute este script novamente"
echo "  - Para limpeza periódica, execute: ./scripts/limpar_metadados_macos.sh"
echo ""
