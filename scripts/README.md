# Scripts de Manutenção do AMPidentifier

Este diretório contém scripts auxiliares para manutenção do repositório.

## Scripts Disponíveis

### atualizar_comparacao_branches.sh

Atualiza automaticamente os documentos de comparação entre branches (beta, main local e origin/main).

**Uso:**
```bash
./scripts/atualizar_comparacao_branches.sh
```

**O que faz:**
- Coleta informações sobre os três branches principais
- Conta commits à frente/atrás entre branches
- Lista arquivos modificados
- Identifica arquivos não commitados
- Gera relatório atualizado em `RESUMO_VISUAL_BRANCHES.md`
- Fornece recomendações baseadas no estado atual

**Quando usar:**
- Após fazer commits em qualquer branch
- Após fazer pull/fetch do repositório remoto
- Antes de fazer merge entre branches
- Sempre que quiser verificar o estado dos branches

**Exemplo de saída:**
```
Atualizando comparação de branches...
Data: 2025-12-29 01:30:00

Informações coletadas:
  Beta: 0d68c57
  Main Local: 6904cdb (+3 commits desde beta)
  Origin/Main: b4d0192 (+9 commits desde beta)
  Arquivos modificados (beta→main): 3
  Arquivos modificados (main→origin): 15
  Arquivos não commitados: 65

Documentos atualizados com sucesso!
```

---

## Adicionando Novos Scripts

Ao adicionar novos scripts neste diretório:

1. Use a extensão `.sh` para scripts bash
2. Adicione shebang no início: `#!/bin/bash`
3. Torne o script executável: `chmod +x scripts/seu_script.sh`
4. Documente o script neste README
5. Adicione comentários explicativos no código

---

## configurar_hd_externo.sh

Configura o HD externo para prevenir a criação de arquivos de metadados do macOS (._*, .DS_Store, etc).

**Uso:**
```bash
./scripts/configurar_hd_externo.sh [caminho_do_volume]
```

**Exemplo:**
```bash
./scripts/configurar_hd_externo.sh /Volumes/promethion
```

**O que faz:**
- Remove todos os arquivos `._*` existentes
- Remove todos os arquivos `.DS_Store`
- Remove diretórios `.Spotlight-V100`, `.Trashes`, `.fseventsd`
- Desabilita indexação do Spotlight no volume
- Configura macOS para não criar `.DS_Store` em volumes USB/rede
- Cria marcadores `.metadata_never_index` e `.noindex`

**Quando usar:**
- Após conectar o HD externo pela primeira vez
- Após conectar o HD em um Mac diferente
- Quando notar que arquivos `._*` estão sendo criados novamente

---

## limpar_metadados_macos.sh

Script rápido para limpar arquivos de metadados do macOS em qualquer diretório.

**Uso:**
```bash
./scripts/limpar_metadados_macos.sh [caminho]
```

**Exemplo:**
```bash
# Limpar diretório atual
./scripts/limpar_metadados_macos.sh

# Limpar diretório específico
./scripts/limpar_metadados_macos.sh /Volumes/promethion/AMPidentifier
```

**O que faz:**
- Conta quantos arquivos `._*` e `.DS_Store` existem
- Remove todos esses arquivos
- Reporta quantos arquivos foram removidos

**Quando usar:**
- Antes de fazer commit no Git
- Após desconectar e reconectar o HD externo
- Periodicamente para manutenção

---

## run_explainability_analysis.sh

Executa análise completa de explicabilidade usando SHAP para todos os três modelos de predição de AMPs.

**Uso:**
```bash
./scripts/run_explainability_analysis.sh
```

**O que faz:**
- Verifica se os modelos foram treinados
- Instala dependências necessárias (SHAP, matplotlib, seaborn)
- Executa análise SHAP para RF, SVM e GB
- Gera visualizações abrangentes (summary plots, bar plots, waterfall plots, dependence plots)
- Cria tabelas de importância de features
- Gera relatório Markdown completo
- Cria gráfico de comparação entre modelos

**Quando usar:**
- Após treinar os modelos
- Para gerar relatórios de explicabilidade
- Para entender quais features são mais importantes
- Para demonstrar que os modelos não são caixas pretas

**Saída:**
- Diretório: `model_training/explainability_reports/`
- 3 summary plots (um por modelo)
- 3 bar plots (um por modelo)
- 9 waterfall plots (3 por modelo)
- 15 dependence plots (5 por modelo)
- 3 tabelas CSV de importância
- 1 gráfico de comparação entre modelos
- 1 relatório Markdown completo

**Tempo estimado:** 10-15 minutos

---

## Prevenindo Arquivos de Metadados do macOS

### Problema

O macOS cria automaticamente arquivos de metadados:
- `._*` - Armazenam atributos estendidos e resource forks
- `.DS_Store` - Armazenam configurações de visualização de pastas
- `.Spotlight-V100` - Índice do Spotlight
- `.Trashes` - Lixeira do volume

Esses arquivos são problemáticos em:
- Repositórios Git
- HDs externos compartilhados
- Sistemas de arquivos não-macOS

### Solução Implementada

1. **`.gitignore` atualizado** - Ignora esses arquivos no Git
2. **Scripts de limpeza** - Remove arquivos existentes
3. **Configuração do macOS** - Previne criação futura
4. **Marcadores de volume** - Desabilita indexação

### Configuração Global do macOS

Os scripts já aplicam essas configurações, mas você pode verificar:

```bash
# Verificar configurações atuais
defaults read com.apple.desktopservices DSDontWriteNetworkStores
defaults read com.apple.desktopservices DSDontWriteUSBStores

# Aplicar manualmente (já feito pelos scripts)
defaults write com.apple.desktopservices DSDontWriteNetworkStores -bool true
defaults write com.apple.desktopservices DSDontWriteUSBStores -bool true
```

### Workflow Recomendado

1. **Primeira vez:** Execute `./scripts/configurar_hd_externo.sh`
2. **Antes de commits:** Execute `./scripts/limpar_metadados_macos.sh`
3. **Após reconectar HD:** Execute `./scripts/limpar_metadados_macos.sh`

---

## Convenções

- Scripts devem ser executados a partir do diretório raiz do projeto
- Use caminhos relativos quando possível
- Sempre verifique se comandos críticos existem antes de executá-los
- Forneça mensagens de erro claras
- Use `set -e` para parar em caso de erro
