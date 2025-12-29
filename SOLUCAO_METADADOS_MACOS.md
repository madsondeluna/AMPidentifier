# Solução: Arquivos de Metadados do macOS

**Data:** 2025-12-29  
**Problema:** Arquivos `._*` e `.DS_Store` sendo criados no HD externo

---

## Problema Identificado

O macOS cria automaticamente arquivos de metadados em volumes externos:
- **348 arquivos `._*`** foram encontrados inicialmente
- Esses arquivos armazenam atributos estendidos e resource forks
- Causavam erros no Git: `error: non-monotonic index`
- Poluíam o repositório e ocupavam espaço desnecessário

---

## Solução Implementada

### 1. Limpeza Imediata

- Removidos todos os 348 arquivos `._*`
- Removidos todos os arquivos `.DS_Store`
- Removidos diretórios `.Spotlight-V100`, `.Trashes`, `.fseventsd`

### 2. Prevenção Futura

**Arquivo `.gitignore` atualizado:**
```gitignore
# OS
.DS_Store
._*
.Spotlight-V100
.Trashes
Thumbs.db
```

**Configurações do macOS aplicadas:**
```bash
# Desabilitar criação de .DS_Store em volumes de rede
defaults write com.apple.desktopservices DSDontWriteNetworkStores -bool true

# Desabilitar criação de .DS_Store em volumes USB
defaults write com.apple.desktopservices DSDontWriteUSBStores -bool true
```

**Marcadores criados no volume:**
- `.metadata_never_index` - Desabilita indexação do Spotlight
- `.noindex` - Previne indexação adicional

### 3. Scripts de Manutenção

Dois scripts foram criados para facilitar a manutenção:

**`scripts/configurar_hd_externo.sh`**
- Configuração completa do HD externo
- Remove arquivos existentes
- Aplica todas as configurações de prevenção
- Uso: `./scripts/configurar_hd_externo.sh /Volumes/promethion`

**`scripts/limpar_metadados_macos.sh`**
- Limpeza rápida de metadados
- Pode ser executado em qualquer diretório
- Reporta quantos arquivos foram removidos
- Uso: `./scripts/limpar_metadados_macos.sh`

---

## Workflow Recomendado

### Primeira Vez (Já Executado)
```bash
./scripts/configurar_hd_externo.sh /Volumes/promethion
```

### Manutenção Regular

**Antes de fazer commits:**
```bash
./scripts/limpar_metadados_macos.sh
git status
```

**Após desconectar e reconectar o HD:**
```bash
./scripts/limpar_metadados_macos.sh
```

**Se conectar em outro Mac:**
```bash
./scripts/configurar_hd_externo.sh /Volumes/promethion
```

---

## Resultados

### Antes
- 348 arquivos `._*` no repositório
- Erros constantes do Git
- Arquivos de metadados sendo rastreados

### Depois
- 0 arquivos `._*` no repositório
- `.gitignore` configurado corretamente
- Configurações do macOS aplicadas
- Scripts de manutenção disponíveis
- Prevenção automática ativada

---

## Verificação

Para verificar se tudo está funcionando:

```bash
# Verificar se não há arquivos ._*
find . -name "._*" -type f

# Verificar configurações do macOS
defaults read com.apple.desktopservices DSDontWriteNetworkStores
defaults read com.apple.desktopservices DSDontWriteUSBStores

# Verificar .gitignore
grep "._\*" .gitignore

# Verificar marcadores no volume
ls -la /Volumes/promethion/ | grep -E "(metadata_never_index|noindex)"
```

---

## Notas Importantes

1. **Configurações são por Mac:** Se você conectar o HD em outro Mac, execute o script de configuração novamente.

2. **Arquivos podem reaparecer:** Algumas operações do macOS podem criar esses arquivos. Use o script de limpeza periodicamente.

3. **Git não rastreia mais:** O `.gitignore` garante que novos arquivos `._*` não sejam adicionados ao repositório.

4. **Limpeza automática:** Considere adicionar o script de limpeza a um hook do Git (pre-commit).

---

## Próximos Passos Opcionais

### Adicionar Hook Pre-Commit

Para limpeza automática antes de cada commit:

```bash
# Criar arquivo .git/hooks/pre-commit
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
./scripts/limpar_metadados_macos.sh > /dev/null 2>&1
EOF

# Tornar executável
chmod +x .git/hooks/pre-commit
```

### Adicionar ao Cron (Limpeza Periódica)

Para limpeza automática diária:

```bash
# Editar crontab
crontab -e

# Adicionar linha (executa todo dia às 2h da manhã)
0 2 * * * cd /Volumes/promethion/AMPidentifier && ./scripts/limpar_metadados_macos.sh >> /tmp/cleanup.log 2>&1
```

---

## Referências

- [Apple Developer: File System Events](https://developer.apple.com/library/archive/documentation/Darwin/Conceptual/FSEvents_ProgGuide/)
- [macOS .DS_Store Documentation](https://en.wikipedia.org/wiki/.DS_Store)
- [Resource Forks and AppleDouble](https://en.wikipedia.org/wiki/AppleSingle_and_AppleDouble_formats)
