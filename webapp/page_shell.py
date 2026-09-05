"""Shell compartilhado das paginas Pure Design servidas sob /beta.

Uma so fonte para o que as tres paginas dividem: o head, a folha de
estilo, a barra fixa de cima, a barra fixa de baixo e o modal de
feedback. Pagina nova fornece o proprio miolo e o proprio javascript, e
nao copia nada disto: markup repetido em tres arquivos diverge no
segundo commit.

O head e gabarito de %(nome)s, nao de Jinja: as chaves duplas do Jinja
continuam intactas e sao resolvidas depois, por render_template_string.
"""

# ---------------------------------------------------------------------------
# Copia do shell nos dois idiomas.
#
# Nao se traduz: nome da marca, nome de instituicao, sigla de metrica,
# codigo de sequencia, rotulo de rota tecnica como /health. Sao registros,
# e traduzir registro quebra o casamento com a fonte que o declara.
# ---------------------------------------------------------------------------

LANGS = ('en', 'pt')

STRINGS = {
    'nav_predict':      {'en': 'Predict',            'pt': 'Prever'},
    'nav_about':        {'en': 'About',              'pt': 'Sobre'},
    'nav_suggestions':  {'en': 'Suggestions',        'pt': 'Sugestões'},
    'nav_beta':         {'en': 'Beta Version',       'pt': 'Versão beta'},
    'nav_home':         {'en': 'AMPidentifier, home', 'pt': 'AMPidentifier, início'},
    'nav_label':        {'en': 'Main',               'pt': 'Principal'},
    'skip':             {'en': 'Skip to content',    'pt': 'Ir para o conteúdo'},

    'status_checking':  {'en': 'Checking',           'pt': 'Verificando'},
    'status_online':    {'en': 'Online: model loaded, predictions ready',
                         'pt': 'Online: modelo carregado, predições prontas'},
    'status_offline':   {'en': 'Offline: backend unreachable, try again shortly',
                         'pt': 'Offline: servidor inalcançável, tente em instantes'},
    'status_wait':      {'en': 'Checking server status',
                         'pt': 'Verificando o estado do servidor'},
    'status_latency':   {'en': '(Xms) = current /health round-trip latency',
                         'pt': '(Xms) = latência atual de ida e volta em /health'},

    'src_github':       {'en': 'Source on GitHub',   'pt': 'Código no GitHub'},
    'src_pypi':         {'en': 'Package on PyPI',    'pt': 'Pacote no PyPI'},
    'mode_to_dark':     {'en': 'Switch to dark mode', 'pt': 'Mudar para o modo escuro'},
    'mode_to_light':    {'en': 'Switch to light mode', 'pt': 'Mudar para o modo claro'},
    'slogan':           {'en': 'antimicrobial peptide prediction',
                         'pt': 'predição de peptídeos antimicrobianos'},
    'lang_switch':      {'en': 'Ver em português',   'pt': 'Read in English'},

    'group_inst':       {'en': 'Institutions',       'pt': 'Instituições'},
    'group_dept':       {'en': 'Departments',        'pt': 'Departamentos'},
    'group_fund':       {'en': 'Funding',            'pt': 'Financiamento'},
    'group_labs':       {'en': 'Research groups',    'pt': 'Grupos de pesquisa'},

    'modal_title':      {'en': 'Report issue or suggestion',
                         'pt': 'Relatar problema ou sugestão'},
    'modal_type':       {'en': 'Type',               'pt': 'Tipo'},
    'modal_bug':        {'en': 'Bug report',         'pt': 'Relato de defeito'},
    'modal_feature':    {'en': 'Feature request',    'pt': 'Pedido de funcionalidade'},
    'modal_other':      {'en': 'Other',              'pt': 'Outro'},
    'modal_desc':       {'en': 'Description',        'pt': 'Descrição'},
    'modal_placeholder':{'en': 'Describe the issue or your suggestion...',
                         'pt': 'Descreva o problema ou a sua sugestão...'},
    'modal_cancel':     {'en': 'Cancel',             'pt': 'Cancelar'},
    'modal_submit':     {'en': 'Open on GitHub',     'pt': 'Abrir no GitHub'},
}


def t(key, lang):
    return STRINGS[key][lang]


# A rota em portugues e a mesma com prefixo. A raiz e o unico caso
# especial: /pt e nao /pt/.
def localised(path, lang):
    if lang == 'en':
        return path
    return '/pt' if path == '/' else '/pt' + path


HEAD = """<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="#ffffff">
<title>%(title)s</title>
<meta name="description" content="%(description)s">
<meta name="keywords" content="antimicrobial peptide prediction, antimicrobial peptide predictor, antimicrobial peptide classifier, antimicrobial peptide classification, antimicrobial peptide identification, antimicrobial peptide detection, antimicrobial peptide screening, antimicrobial peptide discovery, antimicrobial peptide annotation, antimicrobial peptide mining, antimicrobial peptide search, antimicrobial activity prediction, peptide bioactivity prediction, peptide function prediction, bioactive peptide prediction, in silico peptide screening, virtual screening peptides, high throughput peptide screening, AMP prediction, AMP predictor, AMP classifier, AMP identification, AMP detection, AMP screening, AMP discovery, AMP annotation, AMP prediction tool, AMP prediction software, AMP prediction server, AMP prediction web server, AMP prediction online, AMP prediction free, AMP prediction API, AMP prediction pipeline, AMP prediction benchmark, AMP prediction accuracy, AMP prediction machine learning, AMP prediction deep learning, AMP finder, AMP scanner, AMP toolkit, machine learning antimicrobial peptides, deep learning antimicrobial peptides, machine learning bioinformatics, deep learning bioinformatics, ensemble learning, ensemble model, soft voting classifier, stacking ensemble, gradient boosting, extreme gradient boosting, XGBoost, LightGBM, random forest, support vector machine, logistic regression, neural network, multilayer perceptron, convolutional neural network, recurrent neural network, BiLSTM, transformer protein model, protein language model, PLLM, embeddings, feature engineering, feature selection, cross validation, hyperparameter tuning, class imbalance, ROC AUC, sensitivity specificity, Matthews correlation coefficient, confusion matrix, model interpretability, SHAP, amino acid composition, AAC, dipeptide composition, DPC, pseudo amino acid composition, CTD descriptors, composition transition distribution, physicochemical descriptors, net charge, hydrophobicity, hydrophobic moment, isoelectric point, aliphatic index, instability index, molecular weight, helical wheel, amphipathicity, peptide length, sequence descriptors, protein descriptors, iFeature, propy, peptides package, bioinformatics tool, bioinformatics web server, computational biology, structural bioinformatics, proteomics, peptidomics, genomics, transcriptomics, metagenomics, immunoinformatics, molecular biology software, sequence analysis, FASTA, FASTA input, multi FASTA, batch prediction, CSV export, command line interface, CLI tool, Python package, pip install ampidentifier, open source bioinformatics, reproducible research, Google Colab notebook, antibiotic resistance, antimicrobial resistance, AMR, multidrug resistant bacteria, superbugs, ESKAPE pathogens, novel antibiotics, antibiotic alternatives, drug discovery, peptide drug design, therapeutic peptides, host defense peptides, innate immunity, defensins, cathelicidins, bacteriocins, lantibiotics, cecropins, magainins, LL-37, antibacterial peptides, antifungal peptides, antiviral peptides, antiparasitic peptides, antibiofilm peptides, anticancer peptides, cell penetrating peptides, hemolytic activity, cytotoxicity prediction, minimum inhibitory concentration, MIC prediction, plant antimicrobial peptides, insect antimicrobial peptides, marine antimicrobial peptides, APD3, DBAASP, DRAMP, CAMPR3, LAMP database, UniProt, SwissProt, AMP database, curated peptide dataset, training dataset, benchmark dataset, AMPidentifier, ampidentifier, AMPidentifier web, AMPidentifier CLI, AMPidentifier Python, free AMP prediction tool, online AMP predictor, no login bioinformatics tool, AMP prediction 2026, new AMP prediction tool, predição de peptídeos antimicrobianos, peptídeo antimicrobiano, identificação de peptídeos antimicrobianos, classificador de peptídeos antimicrobianos, ferramenta de bioinformática, aprendizado de máquina, aprendizado profundo, resistência antimicrobiana, resistência a antibióticos, descoberta de fármacos, análise de sequências, peptídeos bioativos, peptídeos de defesa, ferramenta gratuita online, 抗菌肽预测, 抗菌肽, 抗菌肽识别, 抗菌肽分类器, 生物信息学工具, 机器学习, 深度学习, 抗生素耐药性, 药物发现, 序列分析, 在线预测工具, 免费工具, रोगाणुरोधी पेप्टाइड भविष्यवाणी, रोगाणुरोधी पेप्टाइड, पेप्टाइड वर्गीकरण, जैव सूचना विज्ञान उपकरण, मशीन लर्निंग, डीप लर्निंग, एंटीबायोटिक प्रतिरोध, दवा खोज, अनुक्रम विश्लेषण, मुफ्त ऑनलाइन उपकरण">
<meta name="author" content="Madson Aragao">
<meta name="robots" content="index, follow">
<link rel="canonical" href="https://www.ampidentifier.com%(path)s">
<meta property="og:locale" content="en_US">
<meta property="og:locale:alternate" content="pt_BR">
<meta property="og:locale:alternate" content="zh_CN">
<meta property="og:locale:alternate" content="hi_IN">
<meta property="og:type" content="website">
<meta property="og:url" content="https://www.ampidentifier.com%(path)s">
<meta property="og:title" content="AMPidentifier | Antimicrobial Peptide Prediction Tool">
<meta property="og:description" content="Free web tool for antimicrobial peptide (AMP) prediction using machine learning ensemble models. Submit FASTA sequences, get predictions in seconds.">
<meta property="og:image" content="https://www.ampidentifier.com/img/og-image.png?v=2">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:url" content="https://www.ampidentifier.com/">
<meta name="twitter:title" content="AMPidentifier | Antimicrobial Peptide Prediction Tool">
<meta name="twitter:description" content="Free web tool for antimicrobial peptide (AMP) prediction using machine learning ensemble models. Submit FASTA sequences, get predictions in seconds.">
<meta name="twitter:image" content="https://www.ampidentifier.com/img/og-image.png?v=2">
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "AMPidentifier",
  "alternateName": ["AMP identifier", "AMPidentifier web server", "AMPidentifier Python package", "Preditor de peptídeos antimicrobianos", "抗菌肽预测工具", "रोगाणुरोधी पेप्टाइड भविष्यवक्ता"],
  "url": "https://www.ampidentifier.com/beta",
  "description": "AMPidentifier is an ensemble machine learning toolkit for antimicrobial peptide (AMP) prediction. It accepts FASTA sequences and returns AMP classification scores using gradient boosting, XGBoost, LightGBM, and a soft-voting ensemble model.",
  "applicationCategory": "Scientific Software",
  "operatingSystem": "Web",
  "inLanguage": ["en", "pt-BR", "zh", "hi"],
  "offers": { "@type": "Offer", "price": "0", "priceCurrency": "USD" },
  "author": {
    "@type": "Person",
    "name": "Madson Aragao",
    "url": "https://github.com/madsondeluna"
  },
  "codeRepository": "https://github.com/madsondeluna/AMPidentifier",
  "license": "https://github.com/madsondeluna/AMPidentifier/blob/main/LICENSE",
  "keywords": ["antimicrobial peptide prediction", "antimicrobial peptide predictor", "antimicrobial peptide classifier", "antimicrobial peptide classification", "antimicrobial peptide identification", "antimicrobial peptide detection", "antimicrobial peptide screening", "antimicrobial peptide discovery", "antimicrobial peptide annotation", "antimicrobial peptide mining", "antimicrobial peptide search", "antimicrobial activity prediction", "peptide bioactivity prediction", "peptide function prediction", "bioactive peptide prediction", "in silico peptide screening", "virtual screening peptides", "high throughput peptide screening", "AMP prediction", "AMP predictor", "AMP classifier", "AMP identification", "AMP detection", "AMP screening", "AMP discovery", "AMP annotation", "AMP prediction tool", "AMP prediction software", "AMP prediction server", "AMP prediction web server", "AMP prediction online", "AMP prediction free", "AMP prediction API", "AMP prediction pipeline", "AMP prediction benchmark", "AMP prediction accuracy", "AMP prediction machine learning", "AMP prediction deep learning", "AMP finder", "AMP scanner", "AMP toolkit", "machine learning antimicrobial peptides", "deep learning antimicrobial peptides", "machine learning bioinformatics", "deep learning bioinformatics", "ensemble learning", "ensemble model", "soft voting classifier", "stacking ensemble", "gradient boosting", "extreme gradient boosting", "XGBoost", "LightGBM", "random forest", "support vector machine", "logistic regression", "neural network", "multilayer perceptron", "convolutional neural network", "recurrent neural network", "BiLSTM", "transformer protein model", "protein language model", "PLLM", "embeddings", "feature engineering", "feature selection", "cross validation", "hyperparameter tuning", "class imbalance", "ROC AUC", "sensitivity specificity", "Matthews correlation coefficient", "confusion matrix", "model interpretability", "SHAP", "amino acid composition", "AAC", "dipeptide composition", "DPC"],
  "isAccessibleForFree": true,
  "softwareVersion": "1.0",
  "installUrl": "https://pypi.org/project/ampidentifier/",
  "sameAs": ["https://github.com/madsondeluna/AMPidentifier", "https://pypi.org/project/ampidentifier/"],
  "citation": "de Luna-Aragao, M. A. et al. (2026). AMPidentifier: A Cross-Platform Ensemble Toolkit for Antimicrobial Peptide Prediction."
}
</script>
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {"@type": "Question", "name": "What is AMPidentifier?",
     "acceptedAnswer": {"@type": "Answer", "text": "AMPidentifier is a free web tool and Python package that predicts whether a peptide sequence is an antimicrobial peptide (AMP) using an ensemble of gradient boosting, XGBoost and LightGBM models."}},
    {"@type": "Question", "name": "How do I predict antimicrobial peptides online?",
     "acceptedAnswer": {"@type": "Answer", "text": "Paste one or more sequences in FASTA format at https://www.ampidentifier.com/beta and submit. The tool returns an AMP probability score for each sequence in seconds, with no login required."}},
    {"@type": "Question", "name": "How accurate is AMPidentifier?",
     "acceptedAnswer": {"@type": "Answer", "text": "On the held-out test set the soft-voting ensemble reaches AUC 0.950, sensitivity 0.949 and specificity 0.784."}},
    {"@type": "Question", "name": "Can I run AMPidentifier locally or in batch?",
     "acceptedAnswer": {"@type": "Answer", "text": "Yes. Install it with pip install ampidentifier and use the command line interface, or clone the GitHub repository. The web version also accepts CSV export of results."}},
    {"@type": "Question", "name": "Which types of antimicrobial peptides does it cover?",
     "acceptedAnswer": {"@type": "Answer", "text": "The model is trained on experimentally validated AMPs from public databases, covering antibacterial, antifungal, antiviral and other host defense peptides, against non-AMP sequences."}},
    {"@type": "Question", "name": "How does AMPidentifier differ from other AMP prediction tools?",
     "acceptedAnswer": {"@type": "Answer", "text": "It combines gradient boosting, XGBoost and LightGBM in a soft-voting ensemble, runs as a web server, a command line tool and a Python package, and exposes the full training code and datasets on GitHub."}},
    {"@type": "Question", "name": "Are submitted sequences stored?",
     "acceptedAnswer": {"@type": "Answer", "text": "No. Sequences are processed in memory and are not stored on the server."}}
  ]
}
</script>
<link rel="icon" type="image/svg+xml" href="/img/symbol-fan.svg">
<link rel="apple-touch-icon" href="/img/symbol-fan.svg">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@125,300..400&family=Public+Sans:wght@300;400;500;600&family=Spline+Sans+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/pure/tokens.css?v={{ asset_v }}">
<link rel="stylesheet" href="/pure/patterns.css?v={{ asset_v }}">
<link rel="stylesheet" href="/pure/motion.css?v={{ asset_v }}">
<link rel="stylesheet" href="/pure/light.css?v={{ asset_v }}">"""

STYLE = """  /* =========================================================
     Pure Design 1.4.2. Nenhum literal de cor, tipo, raio, sombra
     ou curva abaixo desta linha: falta de valor vira token em
     pure/tokens.css, nunca literal aqui.

     Diagramacao: uma coluna, uma largura, um eixo esquerdo. Todo
     bloco comeca e termina na mesma borda, do cabecalho ao rodape.
     Ritmo da pagina em tres degraus (24, 48, 96), sentence case em
     tudo, um tamanho por classe de controle.

     Componente que ja existe em pure/patterns.css entra por import,
     nao por copia: .num, .tip, .empty, .hit, .field-error, .skeleton.
     O que fica aqui e so posicionamento e o que e proprio da pagina.
     ========================================================= */

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: var(--font-sans);
    font-size: var(--text-15);
    line-height: var(--leading-normal);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: var(--space-48) 0 var(--space-48);
    -webkit-tap-highlight-color: var(--tap-highlight);
  }

  /* uma coluna, uma largura, uma borda esquerda: todo bloco da pagina
     comeca e termina no mesmo lugar, do cabecalho ao rodape. */
  .shell { max-width: var(--container-md); margin: 0 auto; padding: 0 var(--space-24); }
  .shell > * { min-width: 0; }
  .block > * { min-width: 0; }

  .step-1 { margin-top: var(--space-24); }
  .step-2 { margin-top: var(--space-48); }
  .step-3 { margin-top: var(--space-96); }

  h1, h2, h3 {
    font-family: var(--font-display);
    font-stretch: var(--font-display-stretch);
    font-weight: var(--weight-light);
    letter-spacing: var(--tracking-display);
    line-height: var(--leading-tight);
    color: var(--text);
  }
  h2 { font-size: var(--text-display-section); }
  h3 { font-size: var(--text-24); }

  a { color: var(--muted); text-decoration: underline; text-underline-offset: var(--space-2); }
  a:hover { color: var(--text); }

  /* texto justificado sem hifenizacao abre rios: para fechar a margem, a
     linha estica o espaco entre as palavras, e a linha do registro INPI
     chegava a quase o dobro do espaco normal. Hifenizar devolve o espaco
     de palavra ao tamanho de sempre, que e o que justifica justificar. */
  .prose-justify { -webkit-hyphens: auto; hyphens: auto; text-wrap: pretty; }

  /* ---------- cabecalho ---------- */

  .brand-logo { height: var(--space-40); width: auto; display: block; }
  .header-row { display: flex; align-items: center; justify-content: space-between; gap: var(--space-24); flex-wrap: wrap; }

  /* ---------- estado do servidor ---------- */

  .status-row { display: flex; align-items: center; margin-top: var(--space-12); }
  /* .tip cuida de position, opacidade e revelacao no hover e no foco;
     aqui fica so o arranjo da linha e a ancoragem da caixa. */
  .status-tip { display: inline-flex; align-items: center; gap: var(--space-6); }
  .status-label { font-family: var(--font-mono); font-size: var(--text-12); color: var(--muted); transition: color var(--duration-4) var(--ease-standard); }
  .status-label.online  { color: var(--status-good); }
  .status-label.offline { color: var(--status-critical); }

  .status-dot {
    width: var(--space-8);
    height: var(--space-8);
    border-radius: var(--radius-circle);
    background: var(--border);
    flex-shrink: 0;
    transition: background-color var(--duration-4) var(--ease-standard);
  }
  .status-dot.online  { background: var(--status-good); }
  .status-dot.offline { background: var(--status-critical); }

  .status-tip [role="tooltip"] {
    position: absolute;
    left: calc(100% + var(--space-8));
    top: 50%;
    transform: translateY(-50%);
    display: grid;
    gap: var(--space-4);
    padding: var(--space-8) var(--space-12);
    border-radius: var(--radius-field);
    background: var(--surface);
    border: var(--hairline) solid var(--border);
    color: var(--text);
    font-size: var(--text-12);
    line-height: var(--leading-snug);
    white-space: nowrap;
    z-index: 20;
  }
  .tt-row { display: flex; align-items: center; gap: var(--space-6); }
  .tt-dot { width: var(--space-6); height: var(--space-6); border-radius: var(--radius-circle); flex-shrink: 0; }
  .tt-dot.c-good { background: var(--status-good); }
  .tt-dot.c-crit { background: var(--status-critical); }
  .tt-dot.c-idle { background: var(--border); }
  .tt-note { color: var(--muted); font-family: var(--font-mono); font-size: var(--text-11); }

  /* ---------- texto de abertura e avisos ---------- */

  /* o bloco de abertura nao inventa material: e a mesma construcao das
     faixas de numeros, faixa em --dim com o vidro dentro. Sobre --bg
     branco o vidro sozinho some, e e a faixa que da a ele o que clarear.
     Raio concentrico e cursor default, como a celula .metric. */
  .intro {
    display: flex;
    flex-direction: column;
    gap: var(--space-12);
    padding: var(--space-10) var(--space-12);
    border-radius: var(--radius-field);
    cursor: default;
  }
  .intro p { margin: 0; }

  /* a definicao fica em tinta cheia e a condicao de uso recua para
     --muted: a hierarquia sai do eixo de tinta, nao de um corpo novo,
     que a escala reserva --text-15 para prosa */
  .sub { font-size: var(--text-15); line-height: var(--leading-normal); color: var(--text); }
  .sub strong { font-weight: var(--weight-medium); }

  /* a frase que explicava o CLI era copy de apoio: o que resta e o
     comando e para onde ir. Os grupos se separam por espaco: o ponto
     entre eles era ruido numa linha que ja tem tres marcas de pontuacao */
  .install {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-8);
    font-size: var(--text-13);
    color: var(--muted);
  }
  .install code {
    font-family: var(--font-mono);
    font-size: var(--text-13);
    background: var(--dim);
    color: var(--text);
    padding: var(--space-4) var(--space-10);
    border-radius: var(--radius-mark);
  }
  /* as tres linhas sao o mesmo tipo de informacao, onde instalar e para
     onde ir, entao andam juntas: gap menor entre elas que o do bloco,
     que separa a definicao do grupo inteiro */
  .install-stack { display: flex; flex-direction: column; gap: var(--space-6); }

  /* ---------- numeros do modelo ---------- */

  /* o fundo da pagina e branco chapado e vidro precisa de algo atras
     para refratar: sobre branco, --glass-fill e --glass-edge tambem sao
     brancos e a celula some. A faixa em --dim da as celulas o que
     clarear, e so entao o material aparece. */
  .metrics-band { background: var(--dim); border-radius: var(--radius-surface); padding: var(--space-12); }
  .metrics-label { font-family: var(--font-mono); font-size: var(--text-11); letter-spacing: var(--tracking-wide); color: var(--muted); margin: 0 0 var(--space-10) var(--space-4); }
  /* o changelog e a mesma construcao da faixa de numeros: faixa em --dim
     com o vidro dentro, raio concentrico, cursor default. O que muda e o
     conteudo, prosa em vez de celula. */
  .changelog {
    border-radius: var(--radius-field);
    padding: var(--space-10) var(--space-12);
    display: flex;
    flex-direction: column;
    gap: var(--space-10);
    cursor: default;
  }
  .changelog p { margin: 0; font-size: var(--text-15); line-height: var(--leading-normal); color: var(--text); }
  /* o corpo recua para --muted: a primeira frase diz o escopo em tinta
     cheia e o detalhe tecnico fica um nivel abaixo, no mesmo corpo */
  .changelog .changelog-body { color: var(--muted); }

  .metrics-grid { display: grid; gap: var(--space-8); }
  .metrics-3 { grid-template-columns: repeat(3, 1fr); }
  .metrics-4 { grid-template-columns: repeat(4, 1fr); }
  /* raio concentrico: a celula nunca arredonda mais que a faixa */
  /* o hover de .card-glass fica: clareia o vidro e levanta a sombra, e a
     celula responde ao ponteiro. O cursor continua default, porque a
     celula nao abre nada: o que ela promete e reacao, nao clique. */
  .metric {
    border-radius: var(--radius-field);
    padding: var(--space-10) var(--space-12);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    cursor: default;
  }
  .metric-val { font-size: var(--text-20); line-height: var(--leading-none); color: var(--text); }
  .metric-lbl { font-family: var(--font-mono); font-size: var(--text-11); letter-spacing: var(--tracking-wide); color: var(--muted); }

  /* ---------- compartilhar ---------- */

  .share-section { padding: var(--space-24); }
  .share-inner { display: flex; align-items: center; justify-content: space-between; gap: var(--space-16); flex-wrap: wrap; }
  .share-heading { font-size: var(--text-15); font-weight: var(--weight-medium); color: var(--text); }
  .share-actions { display: flex; gap: var(--space-8); flex-shrink: 0; }

  .share-url-box {
    display: none;
    margin-top: var(--space-12);
    padding: var(--space-8) var(--space-12);
    border-radius: var(--radius-field);
    background: var(--dim);
    border: var(--hairline) solid var(--border);
    font-family: var(--font-mono);
    font-size: var(--text-12);
    color: var(--text);
    word-break: break-all;
  }
  .share-url-box.open { display: block; }
  .share-form { display: none; margin-top: var(--space-12); gap: var(--space-8); align-items: center; flex-wrap: wrap; }
  .share-form .pill { align-self: stretch; }
  .share-form.open { display: flex; }
  /* sem piso explicito: o minimo automatico do item flex ja segura o
     campo no proprio conteudo, e nao ha token de largura para esse papel */
  .share-form .field { flex: 1; }
  .share-form-status { flex-basis: 100%; font-size: var(--text-13); color: var(--muted); min-height: var(--space-16); }

  /* ---------- entrada ---------- */

  .label-row { display: flex; align-items: baseline; justify-content: space-between; gap: var(--space-12); margin-bottom: var(--space-8); }
  .seq-counter { font-family: var(--font-mono); font-size: var(--text-12); color: var(--muted); }

  #fasta { min-height: calc(var(--field-height) * 4); font-family: var(--font-mono); }
  #fileInput { display: none; }

  /* uma linha para tudo que age sobre a entrada: modelo, executar,
     limpar, exemplo e upload comecam na mesma borda do campo acima. */
  .row { display: flex; gap: var(--space-12); margin-top: var(--space-12); align-items: center; flex-wrap: wrap; }
  .row .select-shell { flex: 1; }

  #status { font-size: var(--text-13); color: var(--muted); margin-top: var(--space-12); min-height: var(--space-16); }
  .err { color: var(--status-critical); }

  /* ---------- resultados ---------- */

  #results:empty { display: none; }
  /* o vazio e uma linha, nao um bloco: o texto atravessa a largura e a
     acao fica na ponta, em vez de um retangulo alto com um canto escrito
     e o resto vazio. */
  #results > .empty {
    margin-top: var(--space-24);
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-24);
    padding: var(--space-24);
  }
  #results > .empty .pill { flex-shrink: 0; white-space: nowrap; }
  /* duas linhas: o estado em ink cheio, o proximo passo em --muted logo
     abaixo, na mesma medida do resto da pagina */
  .empty-head { color: var(--text); margin-bottom: var(--space-4); }

  /* o esqueleto copia a caixa final: cartao de resumo com tres numeros e
     as primeiras linhas da tabela, na mesma altura, para a chegada dos
     dados nao empurrar a pagina. */
  .sk-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--space-24); margin-top: var(--space-16); }
  .sk-val { height: var(--space-32); border-radius: var(--radius-mark); }
  .sk-val + .skeleton-line { margin-top: var(--space-6); }
  .sk-rows { margin-top: var(--space-24); }
  .sk-row { height: var(--space-40); border-radius: var(--radius-mark); }
  .sk-row + .sk-row { margin-top: var(--space-8); }

  .summary { padding: var(--space-24); }
  .summary-title { font-size: var(--text-15); font-weight: var(--weight-medium); color: var(--text); }
  .summary-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--space-24); margin-top: var(--space-16); }
  /* o numero fica na tinta do texto. Cor de serie como tipo derruba o
     contraste: --chart-2 a 20px da 3.38:1 e a 13px da 3.66:1, os dois
     abaixo do piso de 4.5. A serie aparece na marca, nunca na letra. */
  .stat-val {
    font-size: var(--text-32);
    line-height: var(--leading-none);
    color: var(--text);
  }
  .stat-label { font-family: var(--font-mono); font-size: var(--text-12); letter-spacing: var(--tracking-wide); color: var(--muted); margin-top: var(--space-6); }

  .filter-row { display: flex; gap: var(--space-8); margin-top: var(--space-24); flex-wrap: wrap; }
  .filter-btn[aria-pressed="true"] { color: var(--text); border-color: var(--text); }

  .table-scroll { width: 100%; overflow-x: auto; margin-top: var(--space-16); -webkit-overflow-scrolling: touch; }
  /* com largura 100% e sem piso, a tabela encolhia junto com a tela e a
     rolagem do proprio container nunca entrava: o ID quebrava a cada doze
     caracteres e uma linha virava cinco. O piso e a largura de leitura. */
  table { width: 100%; min-width: var(--measure-wide); border-collapse: collapse; font-size: var(--text-13); }
  th {
    text-align: left;
    font-family: var(--font-mono);
    font-size: var(--text-12);
    font-weight: var(--weight-regular);
    letter-spacing: var(--tracking-wide);
    color: var(--muted);
    padding: var(--space-8) var(--space-10);
    border-bottom: var(--hairline) solid var(--border);
    white-space: nowrap;
  }
  td {
    padding: var(--space-10);
    border-bottom: var(--hairline) solid var(--border);
    color: var(--text);
    overflow-wrap: anywhere;
  }
  td.seq-id { font-family: var(--font-mono); }

  /* a classe carrega a serie na bolinha e deixa o rotulo na tinta do
     texto, entao a coluna continua legivel de relance e passa no piso.
     Azul e vermelho saem os dois da paleta categorica, --chart-1 e
     --chart-8: o vermelho da serie nao pode ser --status-critical, que e
     cor de estado e nao carrega identidade de serie. O par tambem separa
     melhor que o anterior sob deuteranopia e protanopia. */
  .pred { display: inline-flex; align-items: center; gap: var(--space-8); white-space: nowrap; }
  .pred-dot { width: var(--space-8); height: var(--space-8); border-radius: var(--radius-circle); flex-shrink: 0; }
  .pred-dot.is-amp { background: var(--chart-1); }
  .pred-dot.is-non { background: var(--chart-8); }

  .prob-cell { white-space: nowrap; }
  .prob-bar {
    display: inline-block;
    width: var(--space-56);
    height: var(--space-6);
    background: var(--dim);
    border-radius: var(--radius-mark);
    vertical-align: middle;
    margin-right: var(--space-8);
    overflow: hidden;
  }
  /* o preenchimento cresce por escala, com origem na esquerda */
  .prob-fill {
    display: block;
    height: 100%;
    border-radius: var(--radius-mark);
    transform-origin: left;
    transform: scaleX(var(--fill, 0));
  }
  .prob-fill.is-amp { background: var(--chart-1); }
  .prob-fill.is-non { background: var(--chart-8); }
  /* numero que e comparado entre linhas alinha pela direita, dentro de uma
     caixa de largura fixa: sem isso a virgula decimal anda de linha em
     linha e a coluna so pode ser lida pela barra. */
  .prob-text { display: inline-block; min-width: var(--space-48); text-align: right; font-size: var(--text-12); color: var(--muted); }

  .dl { margin-top: var(--space-16); display: flex; gap: var(--space-8); flex-wrap: wrap; }

  .result-note {
    margin-top: var(--space-24);
    font-size: var(--text-13);
    line-height: var(--leading-normal);
    color: var(--muted);
  }

  .email-csv-section { margin-top: var(--space-24); padding: var(--space-24); }
  .email-csv-title { font-size: var(--text-15); font-weight: var(--weight-medium); color: var(--text); }
  .email-csv-fields { display: grid; grid-template-columns: auto 1fr auto; gap: var(--space-12); align-items: end; margin-top: var(--space-16); }
  /* o campo passa do piso de 40 e fecha em 44: com min-height no piso, o
     botao terminava quatro pixels mais baixo que o campo ao lado, na
     mesma linha. */
  .email-csv-fields .pill { min-height: var(--field-height-touch); }
  .email-csv-status { font-size: var(--text-13); color: var(--muted); margin-top: var(--space-12); min-height: var(--space-16); }
  .email-csv-status.status-good,
  .share-form-status.status-good { color: var(--status-good); }

  /* ---------- mapa de uso ---------- */

  .usage-map-title { font-size: var(--text-15); font-weight: var(--weight-medium); color: var(--text); margin-bottom: var(--space-12); }
  #usageMap { position: relative; border: var(--hairline) solid var(--border); border-radius: var(--radius-surface); background: var(--surface); padding: var(--space-12); }
  /* sem ponto nenhum o mapa e um mundo cinza que parece carregamento
     travado. O vazio e o erro trocam a caixa pelo aviso, e .empty ja traz
     a propria moldura, entao a do bloco sai. */
  #usageMap.is-note { border: 0; padding: 0; background: none; }
  #usageMap svg { display: block; width: 100%; height: auto; }
  #usageMap .land { fill: var(--dim); stroke: var(--border); stroke-width: 0.6; }
  #usageMap .ring {
    fill: var(--secondary);
    fill-opacity: 0.22;
    stroke: var(--secondary);
    stroke-width: 1.3;
    stroke-opacity: 0.78;
    transition: fill-opacity var(--duration-2) var(--ease-standard), stroke-opacity var(--duration-2) var(--ease-standard);
  }
  #usageMap .gloss { fill: url(#ringGloss); pointer-events: none; }
  #usageMap .gloss-in  { stop-color: var(--glass-specular); }
  #usageMap .gloss-mid { stop-color: var(--glass-specular); stop-opacity: 0.25; }
  #usageMap .gloss-out { stop-color: var(--glass-specular); stop-opacity: 0; }
  #usageMap .spot:hover .ring { fill-opacity: 0.36; stroke-opacity: 0.95; }
  /* o anel desenhado tem o tamanho do dado; o alvo do ponteiro tem o
     tamanho da mao. O segundo circulo e invisivel e so recebe o toque. */
  #usageMap .ring-hit { fill: var(--text); fill-opacity: 0; pointer-events: all; }

  .map-tip {
    position: absolute;
    pointer-events: none;
    opacity: 0;
    transform: translate(-50%, -100%);
    padding: var(--space-6) var(--space-10);
    border-radius: var(--radius-field);
    background: var(--surface);
    border: var(--hairline) solid var(--border);
    box-shadow: var(--shadow-glass-rest);
    white-space: nowrap;
    transition: opacity var(--duration-2) var(--ease-standard);
  }
  .map-tip .place { font-size: var(--text-12); color: var(--text); }
  .map-tip .value { font-family: var(--font-mono); font-size: var(--text-11); color: var(--muted); margin-top: var(--space-2); }

  /* ---------- rodape ---------- */

  footer { margin-top: var(--space-48); padding-top: var(--space-24); border-top: var(--hairline) solid var(--border); font-size: var(--text-13); line-height: var(--leading-normal); color: var(--muted); }
  footer p + p { margin-top: var(--space-8); }
  .feedback-link {
    color: var(--muted);
    text-decoration: underline;
    text-underline-offset: var(--space-2);
    cursor: pointer;
    background: none;
    border: none;
    font-family: inherit;
    font-size: inherit;
    padding: 0;
  }
  .feedback-link:hover { color: var(--text); }

  /* metadado e --text-12 mono: a 13 do rodape a mono fica maior que a
     prosa ao lado, porque tem largura e altura de x maiores. */
  .version { font-family: var(--font-mono); font-size: var(--text-12); color: var(--muted); }

  /* uma tira so, as nove marcas lado a lado. O agrupamento em quatro
     titulos deixava metade da largura vazia a direita de cada grupo e
     cobrava quatro rotulos por uma informacao que o alt de cada imagem
     ja carrega. Aqui o rodape encolhe e a leitura vira uma linha. */
  /* nove marcas em uma coluna de 720: o vao entra pequeno e o resto da
     folga abre entre elas, senao a ultima cai para a segunda linha. */
  /* as marcas voltam agrupadas por papel: quatro colunas numa fileira, o
     rotulo em caixa de frase porque caixa alta nao existe na linguagem */
  /* quem quebra e a tira, nunca o grupo: com wrap na fileira interna um
     grupo de tres marcas descia uma para a linha de baixo e o rotulo
     passava a cobrir meia lista. O grupo e indivisivel e as marcas descem
     um degrau de altura, que quatro colunas nao cabem na medida da pagina
     no tamanho da tira unica. */
  .logo-strip { display: flex; flex-wrap: wrap; align-items: flex-start; justify-content: space-between; gap: var(--space-24) var(--space-16); }
  .logo-group { display: flex; flex-direction: column; align-items: flex-start; gap: var(--space-10); }
  .logo-group-label { font-family: var(--font-mono); font-size: var(--text-11); letter-spacing: var(--tracking-wide); color: var(--muted); }
  /* a fileira tem altura propria e as marcas se centram nela: sem isso
     cada grupo fecha na altura do proprio item mais alto e as quatro
     fileiras do rodape saem em quatro linhas de centro diferentes. */
  .logo-row { display: flex; align-items: center; flex-wrap: nowrap; gap: var(--space-6); min-height: var(--space-48); }
  /* altura igual nao e tamanho igual. Os arquivos ja vem cortados na
     propria tinta, entao o que sobra de diferenca e a forma da marca:
     um logotipo de uma linha gasta a altura toda numa fileira de letras,
     um empilhado divide a mesma altura entre marca e legenda e sai com
     metade do tamanho aparente.

     Tres degraus em vez de dois, pela proporcao medida de cada arquivo:
     acima de 2.4 e logotipo de uma linha e fica no piso; entre 1.3 e 2.4
     e marca com legenda curta; abaixo de 1.3 e empilhado de verdade. A
     regua nao e continua porque altura aqui sai da escala de espaco, e
     inventar um degrau entre 32 e 40 seria inventar token. */
  .logo-row img {
    display: block;
    height: var(--space-32);
    width: auto;
    object-fit: contain;
    filter: grayscale(1);
    opacity: 0.6;
    transition: opacity var(--duration-3) var(--ease-standard);
  }
  .logo-row img.logo-lockup  { height: var(--space-40); }
  .logo-row img.logo-stacked { height: var(--space-48); }
  .logo-row img:hover { opacity: 1; }

  /* ---------- modal ---------- */

  /* o fundo da pagina e branco chapado: vidro sobre ele nao aparece, entao
     o veu escurece com a propria tinta do texto e o cartao e solido. */
  .modal-overlay {
    display: none;
    position: fixed;
    inset: 0;
    z-index: 80;
    place-items: center;
    padding: var(--space-24);
    background: color-mix(in srgb, var(--text) 45%, transparent);
  }
  .modal-overlay.open { display: grid; }
  .modal-card { width: 100%; max-width: var(--container-sm); padding: var(--space-32); box-shadow: var(--shadow-glass-hover); }
  .modal-card h2 { font-size: var(--text-24); margin-bottom: var(--space-24); }
  .modal-card .field + .field { margin-top: var(--space-16); }
  .modal-actions { display: flex; gap: var(--space-10); margin-top: var(--space-24); justify-content: flex-end; }

  /* ---------- reacoes ---------- */

  @media (max-width: 768px) {
    body { padding: var(--space-24) 0 var(--space-24); }
    .step-3 { margin-top: var(--space-48); }
    .summary-grid { gap: var(--space-16); }
    /* quebrada em duas linhas, space-between espalha as tres ultimas de
       ponta a ponta e a fileira sai cheia de buraco: com a tira em duas
       linhas o vao volta a ser fixo e o resto sobra no fim. */
    /* quebrada, space-between espalha os grupos restantes de ponta a
       ponta e a fileira sai cheia de buraco: o vao volta a ser fixo */
    .logo-strip { justify-content: flex-start; gap: var(--space-24); }
    .metrics-4 { grid-template-columns: repeat(2, 1fr); }
    .metric { padding: var(--space-12); }
    .metric-val, .stat-val { font-size: var(--text-20); }
    .status-tip [role="tooltip"] { display: none; }
    .share-actions { width: 100%; }
    .share-actions .pill { flex: 1; }
    .share-form .field, .share-form .select-shell, .share-form .pill { width: 100%; flex-basis: 100%; }
    .email-csv-fields { grid-template-columns: 1fr; }
    .email-csv-fields .pill { width: 100%; }
    /* dividindo a linha com quatro botoes o seletor ficava com 90px e
       mostrava so a primeira letra do modelo. Ele toma a linha inteira,
       como ja faz o formulario de compartilhar, e os botoes dividem a de
       baixo. */
    .row .select-shell { flex-basis: 100%; }
    .row .select { width: 100%; }
    .row .pill { flex: 1; }
    .dl .pill { flex: 1; }
  }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      animation-duration: var(--duration-1);
      animation-iteration-count: 1;
      transition-duration: var(--duration-1);
    }
  }

  /* =========================================================
     Camada de chrome fixo. Acrescentada quando /beta deixou de ser
     pagina unica: a navbar da acesso as outras rotas e o rodape carrega
     as marcas em toda pagina, sem repetir markup em tres arquivos.
     ========================================================= */

  /* As duas barras saem do fluxo, entao quem reserva o espaco delas e o
     corpo. As alturas sao token e sao lidas nos dois lugares, para nao
     poderem divergir: mudar a barra e mudar o respiro na mesma linha. */
  :root {
    --chrome-top: var(--space-64);
    --chrome-bottom: var(--space-96);
  }

  body {
    padding: calc(var(--chrome-top) + var(--space-48)) 0
             calc(var(--chrome-bottom) + var(--space-48));
  }

  /* indice abaixo de 80, que e onde mora o veu do modal: barra acima
     dele cobriria o dialogo e o clique de fora deixaria de fechar. */
  /* As barras sao a textura mais fina do material, nao a mais funda.
     Vidro fundo numa faixa de largura total desfoca 56px o tempo todo,
     derruba a taxa de quadros em aparelho de entrada e le como painel
     em vez de moldura. O que a pagina precisa aqui e translucidez leve:
     o conteudo passa por baixo e ainda se ve.

     as duas barras carregam texto, e texto sobre vidro vai na tinta cheia */
  .navbar .status-label { color: var(--text); white-space: nowrap; }

  .navbar, .footer-bar {
    position: fixed;
    left: 0;
    right: 0;
    z-index: 40;
    border-radius: 0;
  }

  /* Desfoque fino deixa o conteudo atravessar, e numa barra que carrega
     texto isso vira ilegibilidade que depende do que passa por baixo. As
     duas ganham tinta propria: translucida o bastante para o conteudo se
     insinuar, opaca o bastante para o rotulo nao depender dele.

     --surface-context declara o que a barra realmente pinta: a varredura
     de contraste anda pela ancestralidade e sem isto resolveria contra
     --bg, aprovando uma pagina que na tela esta errada. */
  /* Duas classes, e nao uma: `.glass:not(.card-glass)` pesa duas e zera o
     background-color do material. Medido: a declaracao de uma classe nao
     chegava a aplicar e a barra continuava sem tinta propria. */
  .navbar.glass, .footer-bar.glass {
    background-color: color-mix(in srgb, var(--surface) 74%, transparent);
    --surface-context: var(--surface);
  }

  .navbar { top: 0; height: var(--chrome-top); border-bottom: var(--hairline) solid var(--border); }

  /* a regra generica de footer da pagina carrega padding-top, e ela pega
     nesta barra tambem: com ele a fita descia 24px e encostava na borda
     de baixo em vez de se centrar. */
  .footer-bar { bottom: 0; height: var(--chrome-bottom); border-top: var(--hairline) solid var(--border); padding: 0; margin: 0; }

  /* a barra acompanha a coluna da pagina: mesma largura, mesma margem
     lateral, mesmo eixo esquerdo do cabecalho ao rodape. */
  /* As duas barras sao chrome e usam a mesma ancora: largura inteira da
     tela com a folga lateral de 24. Assim a esquerda da navbar cai na
     mesma vertical da esquerda do rodape, e a direita das duas tambem.
     Antes a navbar vivia numa coluna de 1280 centrada, o que a deixava
     sem alinhamento nem com a coluna de texto nem com o rodape. */
  .nav-inner, .footer-bar-inner {
    max-width: var(--container-xl);
    height: 100%;
    margin: 0 auto;
    padding: 0 var(--space-24);
    display: flex;
    align-items: center;
    gap: var(--space-16);
  }

  .nav-inner { gap: var(--space-16); }

  .nav-brand { display: flex; align-items: center; text-decoration: none; flex: 0 0 auto; }

  /* O leque nu carrega a marca na barra, sem placa. Ele e a unica coisa
     colorida ali e nao inverte no escuro: inverter uma marca de cor
     devolve a complementar, nao a versao para fundo escuro. A palavra
     nao se repete aqui porque ela abre a pagina no lockup do topo. */
  .nav-fan { display: block; flex: 0 0 auto; height: var(--space-32); width: auto; }

  /* A fileira de rotas nao quebra: rotulo de navegacao que vira duas
     linhas estica a pilula e rompe a altura da barra, que e fixa. Quando
     nao cabe, ela rola; o que nao pode e o rotulo se partir no meio. */
  .nav-links {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    margin-left: var(--space-16);
    /* a fileira nao cede largura: quem cede e a marca, e dentro dela o
       slogan. So quando a marca ja encolheu a zero e a barra ainda nao
       cabe e que a fileira rola. */
    flex: 0 0 auto;
    min-width: 0;
    overflow-x: auto;
    scrollbar-width: none;
  }

  .nav-links::-webkit-scrollbar { display: none; }

  /* Pilula de vidro de verdade, flutuando sobre a barra. A tentativa
     anterior deixou a sombra do vidro num elemento sem fundo e sem borda,
     e o rotulo saia cercado de um borrao: sombra de vidro pede vidro.
     Agora quem paga a sombra e a propria superficie, e .lit-edge volta a
     ser legitima, porque .pill esta na lista de seletores dela. */
  .nav-link {
    font-family: var(--font-sans);
    font-size: var(--text-12);
    line-height: var(--text-16);
    padding: var(--space-6) var(--space-12);
    min-height: var(--hit-min);
    text-decoration: none;
    /* o rotulo nunca quebra: duas linhas esticam a pilula e rompem a
       altura fixa da barra. Sem lugar, a fileira rola. */
    white-space: nowrap;
    flex: 0 0 auto;
    /* --shadow-glass-rest e desenhada para cartao: a queda larga tem
       18px de raio deslocada 6px, e a pilula tem 30px de altura. A
       sombra fica maior que o objeto e le como borrao sob cada rotulo,
       que e a mesma coisa que a linguagem ja registrou para a lente do
       cursor. --shadow-lens e a resposta dela para vidro pequeno: sobra
       o contato de 1px e as camadas internas, que dao corpo sem
       projetar. */
  }

  /* Duas classes, porque `.pill.lit-edge` pesa duas e declara box-shadow
     propria. Com um seletor de uma classe a troca nao aplicava e a queda
     larga continuava embaixo de cada rotulo. Quarta vez que esta
     armadilha aparece nesta folha. */
  .nav-link.pill { box-shadow: var(--shadow-lens); }
  .nav-link.pill:hover { box-shadow: var(--shadow-glass-rest); }

  /* A rota corrente e a unica tingida. Duas pistas e nenhuma so de cor:
     o vidro assume o acento e o rotulo ganha peso, e aria-current diz o
     mesmo para quem nao ve nenhuma das duas. */
  .nav-link[aria-current="page"] {
    font-weight: var(--weight-medium);
    background-image: var(--glass-tint-accent);
    border-color: var(--glass-edge-accent);
  }

  /* o lado direito nao quebra nem encolhe: ele e a ancora da barra, e a
     fileira de rotas e que rola quando falta espaco */
  .nav-side { display: flex; align-items: center; gap: var(--space-8); margin-left: auto; flex: 0 0 auto; }
  .navbar .status-label { white-space: nowrap; }
  .navbar .status-tip { flex: 0 0 auto; }

  /* o aglomerado liquido tem folga propria de 24px para a massa fundida
     caber; dentro de uma barra de 56 isso estoura, entao a folga cai e o
     tamanho da unidade desce junto. */
  .nav-actions { padding: var(--space-4); --liquid-size: var(--space-32); }

  .footer-bar-inner { gap: var(--space-24); }

  /* doze marcas numa linha so nao cabem em tela estreita, e o corpo da
     pagina nunca rola na horizontal: quem rola e a fita. */
  /* Um bloco centrado, nao quatro ilhas coladas nas bordas da tela: com
     space-between numa barra de largura total os grupos ficam a 500px de
     distancia e param de se ler como uma fita so. O vao entre grupos e
     fixo e maior que o vao entre marcas do mesmo grupo, que e o que faz
     o agrupamento aparecer sem precisar de linha divisoria. */
  /* Ancorada nas pontas, como a navbar, e dentro da mesma coluna: assim
     o primeiro grupo cai na vertical do leque e o ultimo na vertical do
     aglomerado. Espalhar assim na largura INTEIRA da tela era o que
     deixava os grupos a 500px um do outro; dentro da coluna limitada a
     mesma regra alinha em vez de dispersar. */
  .footer-strip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-48);
    flex-wrap: nowrap;
    width: 100%;
    overflow-x: auto;
    overscroll-behavior-x: contain;
  }

  /* o rotulo se centra sobre a fileira que ele nomeia, em vez de comecar
     na borda esquerda dela: com grupos de larguras diferentes o rotulo
     alinhado a esquerda parece solto do proprio conteudo */
  .footer-strip .logo-group { flex: 0 0 auto; gap: var(--space-6); align-items: center; }
  .footer-strip .logo-group-label { white-space: nowrap; color: var(--text); opacity: 0.72; }

  /* cada marca leva ao site da propria instituicao. O alvo e a imagem,
     entao o link nao acrescenta caixa: display block e o raio de marca
     so existem para o anel de foco nao sair quadrado num contorno que
     ninguem ve. */
  .logo-link {
    display: block;
    border-radius: var(--radius-mark);
    line-height: 0;
  }

  /* A altura de cada marca sai de area de caixa constante, nao da altura
     igual: h = raiz(A / proporcao), com A fechada para a marca mais
     quadrada cair no degrau mais alto. Altura igual faz um logotipo de
     uma linha gastar tudo numa fileira de letras e um empilhado dividir
     a mesma altura entre marca e legenda, saindo com metade do tamanho
     aparente. Area de CAIXA, nao de tinta: area de tinta mede espessura
     de traco e infla uma marca de traco fino ate ela virar a maior da
     fita. Os tres degraus sao os que a escala de espaco oferece perto
     dos valores calculados, e o calculo esta em cada arquivo.

     Quatro marcas sobem um degrau acima do que a area pede: ICB, FAPEMIG,
     LCM3 e LNCC. Nas quatro, quem fixa a leitura nao e o simbolo, e uma
     legenda em corpo pequeno ao lado ou abaixo dele, e a area da caixa
     nao ve essa legenda. Marca de letreiro grande, como UFMG, FACEPE e
     LGBV, nao precisa do degrau porque a propria letra e o tamanho. */
  .footer-strip .logo-row { min-height: var(--space-40); gap: var(--space-16); }
  .footer-strip .logo-row img { height: var(--space-24); }
  .footer-strip .logo-row img.logo-lockup { height: var(--space-32); }
  .footer-strip .logo-row img.logo-stacked { height: var(--space-40); }

  /* Contorno de controle no modo claro.

     A aresta do vidro e um realce BRANCO, desenhado para vidro sobre
     fundo colorido. A divergencia local desta pagina faz --bg ser branco
     absoluto, entao o realce cai em cima do proprio fundo: medido, o
     contorno da pilula ficava em 1,00 contra a pagina, que nao e pouco
     contraste, e contorno nenhum. No escuro o mesmo realce da 1,47 e o
     botao se le, que e por que o defeito so aparece de um lado.

     --secondary da 3,45 sobre a pagina, acima do piso de 3 para limite
     de controle, e a linguagem ja a declara como cor de borda e nao de
     texto. O escuro fica como esta. */
  :root:not(.dark) .pill,
  :root:not(.dark) .liquid-item,
  :root:not(.dark) .input,
  :root:not(.dark) .textarea,
  :root:not(.dark) .select {
    border-color: var(--secondary);
  }

  /* o pulo de teclado passa por cima da barra de cima */
  .skip-link { z-index: 100; }

  /* ancora sob barra fixa: o alvo para na borda de baixo da navbar em
     vez de escorregar para tras dela. */
  #main, [id] { scroll-margin-top: calc(var(--chrome-top) + var(--space-16)); }

  /* Os dois icones ficam no DOM e trocam por opacidade e escala, nunca
     por display: alternar display nao anima e a caixa do botao pula. Um
     e absoluto sobre o outro, dentro da unidade liquida de tamanho fixo. */
  .mode-btn { position: relative; border: none; background: none; cursor: pointer; }

  /* a sigla e texto dentro de uma unidade de tamanho fixo, e o liquido so
     espelha o que tem tamanho fixo: por isso ela e sigla e nao o nome do
     idioma por extenso */
  .lang-btn {
    font-family: var(--font-mono);
    font-size: var(--text-11);
    letter-spacing: var(--tracking-wide);
    color: var(--text);
    text-decoration: none;
  }

  .mode-btn .icon {
    position: absolute;
    top: 50%;
    left: 50%;
    transition:
      opacity   var(--duration-3) var(--ease-out-expo),
      transform var(--duration-3) var(--ease-out-expo),
      filter    var(--duration-3) var(--ease-out-expo);
    --icon-swap-x: -50%;
    --icon-swap-y: -50%;
    transform: translate(-50%, -50%) scale(1);
  }

  .mode-btn .icon-moon { opacity: 0; transform: translate(-50%, -50%) scale(0.25); filter: blur(var(--motion-blur-2)); }
  .mode-btn .icon-sun  { opacity: 1; filter: blur(0); }

  :root.dark .mode-btn .icon-sun  { opacity: 0; transform: translate(-50%, -50%) scale(0.25); filter: blur(var(--motion-blur-2)); }
  :root.dark .mode-btn .icon-moon { opacity: 1; transform: translate(-50%, -50%) scale(1); filter: blur(0); }

  /* As marcas institucionais sao tinta escura sobre fundo transparente:
     no modo escuro elas desaparecem no proprio fundo. Inverter devolve a
     mesma marca em tinta clara, que e como cada instituicao publica a
     versao monocromatica dela para fundo escuro. */
  :root.dark .footer-strip .logo-row img { filter: grayscale(1) invert(1); }

  :root.dark .footer-strip .logo-row img { opacity: 0.72; }
  :root.dark .footer-strip .logo-row img:hover { opacity: 1; }

  @media (max-width: 768px) {
    .nav-links { margin-left: var(--space-8); gap: 0; }
    .nav-link { padding: var(--space-6); }
    .status-label { display: none; }
  }"""

DEFS = """<svg class="pure-defs" aria-hidden="true" focusable="false" width="0" height="0"><defs>

  <filter id="pure-goo-tight" x="-40%" y="-40%" width="180%" height="180%" color-interpolation-filters="sRGB">
    <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur"/>
    <feColorMatrix in="blur" type="matrix"
      values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 20 -7.83" result="goo"/>
    <feComposite in="SourceGraphic" in2="goo" operator="atop"/>
  </filter>

  <filter id="pure-goo" x="-40%" y="-40%" width="180%" height="180%" color-interpolation-filters="sRGB">
    <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="blur"/>
    <feColorMatrix in="blur" type="matrix"
      values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 18 -7" result="goo"/>
    <feComposite in="SourceGraphic" in2="goo" operator="atop"/>
  </filter>

  <filter id="pure-goo-wide" x="-40%" y="-40%" width="180%" height="180%" color-interpolation-filters="sRGB">
    <feGaussianBlur in="SourceGraphic" stdDeviation="12" result="blur"/>
    <feColorMatrix in="blur" type="matrix"
      values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 16 -6.17" result="goo"/>
    <feComposite in="SourceGraphic" in2="goo" operator="atop"/>
  </filter>

  <!-- ---------- lente ----------

       O vidro desta linguagem desfoca o fundo. A lente o DOBRA, e a
       diferenca e visivel: dentro dela o que esta atras aparece maior e
       entortado, como atraves de vidro grosso de verdade.

       Nada disso e escala em CSS: e amostragem. feDisplacementMap le a
       posicao de cada pixel do fundo num MAPA, onde o canal R carrega o
       deslocamento horizontal, o G o vertical e 128 e o repouso. Um
       degrade de 255 a 0 atravessando o circulo faz cada ponto amostrar
       de mais perto do centro, e amostrar de mais perto E ampliar.

       Sao DOIS passes encadeados, e serem dois e o ponto:

         corpo  plano no miolo e ingreme na beirada, entao o centro
                amplia limpo. Escala 0,52.
         anel   exatamente 128 nos 80 por cento centrais e vertical nas
                pontas. Nao amplia nada: so entorta o que passa rente a
                borda. Escala 0,18.

       As duas escalas sao a leitura "gota", escolhida na tela entre
       cinco: pequena, muito curva, aumento forte no miolo. As outras
       quatro (lupa, vidro grosso, anel, bolha) mudavam so este par e o
       diametro, entao trocar de leitura e trocar dois numeros aqui e um
       token, nunca reescrever o filtro.

       Num mapa so, subir a beirada arrasta o miolo junto. Separados, um
       se ajusta sem o outro.

       Os numeros e os mapas NAO sao tokens, pela mesma razao dos filtros
       goo logo acima: feImage e feDisplacementMap nao leem var(), e um
       token que nada resolve e um token que mente. Os dois mapas saem de
       tools/lens-map.mjs, entao eles sao reproduziveis em vez de string
       opaca colada aqui.

       primitiveUnits objectBoundingBox e o que faz a lente servir a
       qualquer tamanho: as escalas viram fracao da propria caixa, entao
       trocar --light-lens nao pede filtro novo.

       A regiao vai a 200 por cento porque o deslocamento amostra FORA da
       caixa. Sem a folga, a beirada puxa de uma area que o filtro nao
       tem e devolve um entalhe transparente rente ao contorno. -->
  <filter id="pure-lens" x="-50%" y="-50%" width="200%" height="200%"
          primitiveUnits="objectBoundingBox" color-interpolation-filters="sRGB">
    <feImage href="data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22200%22%20height%3D%22200%22%3E%3Cdefs%3E%3ClinearGradient%20id%3D%22r%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%221%22%20y2%3D%220%22%3E%3Cstop%20offset%3D%220%22%20stop-color%3D%22rgb(255%2C0%2C0)%22%2F%3E%3Cstop%20offset%3D%220.06%22%20stop-color%3D%22rgb(190%2C0%2C0)%22%2F%3E%3Cstop%20offset%3D%220.3%22%20stop-color%3D%22rgb(136%2C0%2C0)%22%2F%3E%3Cstop%20offset%3D%220.5%22%20stop-color%3D%22rgb(128%2C0%2C0)%22%2F%3E%3Cstop%20offset%3D%220.7%22%20stop-color%3D%22rgb(120%2C0%2C0)%22%2F%3E%3Cstop%20offset%3D%220.94%22%20stop-color%3D%22rgb(66%2C0%2C0)%22%2F%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22rgb(0%2C0%2C0)%22%2F%3E%3C%2FlinearGradient%3E%3ClinearGradient%20id%3D%22g%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%220%22%20y2%3D%221%22%3E%3Cstop%20offset%3D%220%22%20stop-color%3D%22rgb(0%2C255%2C0)%22%2F%3E%3Cstop%20offset%3D%220.06%22%20stop-color%3D%22rgb(0%2C190%2C0)%22%2F%3E%3Cstop%20offset%3D%220.3%22%20stop-color%3D%22rgb(0%2C136%2C0)%22%2F%3E%3Cstop%20offset%3D%220.5%22%20stop-color%3D%22rgb(0%2C128%2C0)%22%2F%3E%3Cstop%20offset%3D%220.7%22%20stop-color%3D%22rgb(0%2C120%2C0)%22%2F%3E%3Cstop%20offset%3D%220.94%22%20stop-color%3D%22rgb(0%2C66%2C0)%22%2F%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22rgb(0%2C0%2C0)%22%2F%3E%3C%2FlinearGradient%3E%3C%2Fdefs%3E%3Crect%20width%3D%22200%22%20height%3D%22200%22%20fill%3D%22rgb(128%2C128%2C0)%22%2F%3E%3Ccircle%20cx%3D%22100%22%20cy%3D%22100%22%20r%3D%22100%22%20fill%3D%22url(%23r)%22%2F%3E%3Ccircle%20cx%3D%22100%22%20cy%3D%22100%22%20r%3D%22100%22%20fill%3D%22url(%23g)%22%20style%3D%22mix-blend-mode%3Ascreen%22%2F%3E%3C%2Fsvg%3E" x="0" y="0" width="1" height="1" preserveAspectRatio="none" result="lens-body"/>
    <feImage href="data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22200%22%20height%3D%22200%22%3E%3Cdefs%3E%3ClinearGradient%20id%3D%22r%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%221%22%20y2%3D%220%22%3E%3Cstop%20offset%3D%220%22%20stop-color%3D%22rgb(255%2C0%2C0)%22%2F%3E%3Cstop%20offset%3D%220.1%22%20stop-color%3D%22rgb(128%2C0%2C0)%22%2F%3E%3Cstop%20offset%3D%220.9%22%20stop-color%3D%22rgb(128%2C0%2C0)%22%2F%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22rgb(0%2C0%2C0)%22%2F%3E%3C%2FlinearGradient%3E%3ClinearGradient%20id%3D%22g%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%220%22%20y2%3D%221%22%3E%3Cstop%20offset%3D%220%22%20stop-color%3D%22rgb(0%2C255%2C0)%22%2F%3E%3Cstop%20offset%3D%220.1%22%20stop-color%3D%22rgb(0%2C128%2C0)%22%2F%3E%3Cstop%20offset%3D%220.9%22%20stop-color%3D%22rgb(0%2C128%2C0)%22%2F%3E%3Cstop%20offset%3D%221%22%20stop-color%3D%22rgb(0%2C0%2C0)%22%2F%3E%3C%2FlinearGradient%3E%3C%2Fdefs%3E%3Crect%20width%3D%22200%22%20height%3D%22200%22%20fill%3D%22rgb(128%2C128%2C0)%22%2F%3E%3Ccircle%20cx%3D%22100%22%20cy%3D%22100%22%20r%3D%22100%22%20fill%3D%22url(%23r)%22%2F%3E%3Ccircle%20cx%3D%22100%22%20cy%3D%22100%22%20r%3D%22100%22%20fill%3D%22url(%23g)%22%20style%3D%22mix-blend-mode%3Ascreen%22%2F%3E%3C%2Fsvg%3E" x="0" y="0" width="1" height="1" preserveAspectRatio="none" result="lens-rim"/>
    <feDisplacementMap in="SourceGraphic" in2="lens-body" scale="0.52" xChannelSelector="R" yChannelSelector="G" result="lens-warp"/>
    <feDisplacementMap in="lens-warp" in2="lens-rim" scale="0.18" xChannelSelector="R" yChannelSelector="G" result="lens-bent"/>
    <feGaussianBlur in="lens-bent" stdDeviation="0.005"/>
  </filter>

</defs></svg>
<div class="lit-cursor" aria-hidden="true"></div>"""

NAV = """<nav class="navbar glass glass-thin" aria-label="__nav_label__">
  <div class="nav-inner">
    <a class="nav-brand" href="/" aria-label="__nav_home__">
      <img src="/img/symbol-fan.svg" alt="AMPidentifier" class="nav-fan">
    </a>

    <div class="nav-links">
      <a class="nav-link pill lit lit-edge" href="/" data-nav="predict">__nav_predict__</a>
      <a class="nav-link pill lit lit-edge" href="/about" data-nav="about">__nav_about__</a>
      <a class="nav-link pill lit lit-edge" href="/suggestions" data-nav="suggestions">__nav_suggestions__</a>
      <a class="nav-link pill lit lit-edge" href="/beta" data-nav="beta">__nav_beta__</a>
    </div>

    <div class="nav-side">
      <span class="tip status-tip" tabindex="0" aria-describedby="statusTip">
        <span class="status-dot" id="statusDot"></span>
        <span class="status-label" id="statusLabel">__status_checking__</span>
        <span id="statusTip" role="tooltip">
          <span class="tt-row"><span class="tt-dot c-good"></span> __status_online__</span>
          <span class="tt-row"><span class="tt-dot c-crit"></span> __status_offline__</span>
          <span class="tt-row"><span class="tt-dot c-idle"></span> __status_wait__</span>
          <span class="tt-row tt-note">__status_latency__</span>
        </span>
      </span>

      <!-- aglomerado liquido: duas unidades de tamanho fixo, que e o unico
           formato que o material espelha sem medir texto em javascript -->
      <div class="liquid liquid-tight nav-actions">
        <div class="liquid-sheet" aria-hidden="true">
          <span class="liquid-blob"></span>
          <span class="liquid-blob"></span>
          <span class="liquid-blob"></span>
          <span class="liquid-blob"></span>
        </div>
        <div class="liquid-content">
          <a class="liquid-item" href="https://github.com/madsondeluna/AMPidentifier" target="_blank" rel="noopener" aria-label="__src_github__">
            <svg class="icon" aria-hidden="true"><use href="/pure/icons.svg#branch"></use></svg>
          </a>
          <a class="liquid-item" href="https://pypi.org/project/ampidentifier/" target="_blank" rel="noopener" aria-label="__src_pypi__">
            <svg class="icon" aria-hidden="true"><use href="/pure/icons.svg#code"></use></svg>
          </a>
          <a class="liquid-item lang-btn" id="langBtn" href="__lang_href__"
             aria-label="__lang_switch__" hreflang="__lang_other__" lang="__lang_other__">__lang_code__</a>
          <button class="liquid-item mode-btn" type="button" id="modeBtn" aria-label="__mode_to_dark__" aria-pressed="false">
            <svg class="icon icon-sun" aria-hidden="true"><use href="/pure/icons.svg#sun"></use></svg>
            <svg class="icon icon-moon" aria-hidden="true"><use href="/pure/icons.svg#moon"></use></svg>
          </button>
        </div>
      </div>
    </div>
  </div>
</nav>"""

FOOTER_BAR = """<footer class="footer-bar glass glass-thin">
  <div class="footer-bar-inner">
      <!-- a categoria de cada marca sai do alt, que continua completo: o
           rotulo visivel repetia o que a imagem ja diz e cobrava altura -->
      <div class="footer-strip">
        <div class="logo-group">
          <div class="logo-group-label">__group_inst__</div>
          <div class="logo-row">
            <a class="logo-link" href="https://www.ufpe.br" target="_blank" rel="noopener" title="Universidade Federal de Pernambuco"><img src="/img/pure/ufpe.png"     alt="Universidade Federal de Pernambuco" class="logo-lockup"></a>
            <a class="logo-link" href="https://www.ufmg.br" target="_blank" rel="noopener" title="Universidade Federal de Minas Gerais"><img src="/img/pure/ufmg.png"     alt="Universidade Federal de Minas Gerais"></a>
            <a class="logo-link" href="https://upe.br" target="_blank" rel="noopener" title="Universidade de Pernambuco"><img src="/img/pure/upe-logo.png" alt="Universidade de Pernambuco" class="logo-lockup"></a>
            <a class="logo-link" href="https://www.gov.br/lncc/pt-br" target="_blank" rel="noopener" title="Laboratório Nacional de Computação Científica"><img src="/img/pure/lncc.png"     alt="Laboratório Nacional de Computação Científica" class="logo-lockup"></a>
          </div>
        </div>
        <div class="logo-group">
          <div class="logo-group-label">__group_dept__</div>
          <div class="logo-row">
            <a class="logo-link" href="https://www.ufpe.br/dqf" target="_blank" rel="noopener" title="Departamento de Química Fundamental, UFPE"><img src="/img/pure/dqf.png"   alt="Departamento de Química Fundamental, UFPE" class="logo-lockup"></a>
            <a class="logo-link" href="https://www.ufpe.br/dep-genetica" target="_blank" rel="noopener" title="Departamento de Genética, UFPE"><img src="/img/pure/dgen.png" alt="Departamento de Genética, UFPE" class="logo-lockup"></a>
            <a class="logo-link" href="https://www.icb.ufmg.br" target="_blank" rel="noopener" title="Instituto de Ciências Biológicas, UFMG"><img src="/img/pure/icb.png"   alt="Instituto de Ciências Biológicas, UFMG" class="logo-stacked"></a>
            <a class="logo-link" href="https://www.pgbioinfo.icb.ufmg.br" target="_blank" rel="noopener" title="Programa Interunidades de Pós-Graduação em Bioinformática, UFMG"><img src="/img/pure/ppgbioinfo.png" alt="Programa Interunidades de Pós-Graduação em Bioinformática, UFMG" class="logo-stacked"></a>
          </div>
        </div>
        <div class="logo-group">
          <div class="logo-group-label">__group_fund__</div>
          <div class="logo-row">
            <a class="logo-link" href="https://www.facepe.br" target="_blank" rel="noopener" title="FACEPE"><img src="/img/pure/facepe.png"  alt="FACEPE"></a>
            <a class="logo-link" href="https://fapemig.br" target="_blank" rel="noopener" title="FAPEMIG"><img src="/img/pure/fapemig.png" alt="FAPEMIG" class="logo-stacked"></a>
          </div>
        </div>
        <div class="logo-group">
          <div class="logo-group-label">__group_labs__</div>
          <div class="logo-row">
            <a class="logo-link" href="https://lgbv-ufpe.net" target="_blank" rel="noopener" title="Laboratório de Genética e Biotecnologia Vegetal"><img src="/img/pure/lgbv.png" alt="Laboratório de Genética e Biotecnologia Vegetal"></a>
            <img src="/img/pure/lcm3.png" alt="LCM3" class="logo-lockup">
          </div>
        </div>
      </div>
  </div>
</footer>"""

MODAL = """<!-- Feedback modal -->
<div class="modal-overlay motion-scrim" id="feedbackOverlay" onclick="closeFeedbackOutside(event)">
  <div class="modal modal-card surface motion-modal" role="dialog" aria-modal="true" aria-labelledby="feedbackTitle">
    <h2 id="feedbackTitle">__modal_title__</h2>
    <div class="field">
      <label class="field-label" for="feedbackType">__modal_type__</label>
      <span class="select-shell">
        <select class="select" id="feedbackType">
          <option value="bug">__modal_bug__</option>
          <option value="feature">__modal_feature__</option>
          <option value="other">__modal_other__</option>
        </select>
      </span>
    </div>
    <div class="field">
      <label class="field-label" for="feedbackMsg">__modal_desc__</label>
      <textarea class="textarea" id="feedbackMsg" placeholder="__modal_placeholder__"></textarea>
    </div>
    <div class="modal-actions">
      <button class="pill" onclick="closeFeedback()">__modal_cancel__</button>
      <button class="pill glass-accent" onclick="submitFeedback()">__modal_submit__</button>
    </div>
  </div>
</div>"""

SHELL_JS = """/* ---------- modo: o padrao e claro, a escolha persiste ----------
   Toda pagina abre no claro e em ingles. A preferencia do sistema NAO
   decide: a pagina foi desenhada e medida no claro, e as marcas do rodape
   sao tinta escura invertida por filtro no escuro.

   Depois que a pessoa escolhe, a escolha vale nas paginas seguintes. Ela
   mora em dois lugares e a ordem importa: o endereco vence, porque um
   link compartilhado tem de abrir do jeito que quem mandou estava vendo;
   o armazenamento local entra so quando o endereco nao diz nada. Sem o
   local, trocar de rota pela navbar perderia a escolha; sem o endereco,
   um link compartilhado abriria no modo de quem recebe.

   O idioma nao precisa deste mecanismo: ele mora no caminho, e os links
   da navbar em portugues ja apontam para as rotas em portugues. */

const MODE_KEY = 'amp-mode';
const themeMeta = document.querySelector('meta[name="theme-color"]');
const modeProbe = document.createElement('div');
modeProbe.style.cssText = 'position:absolute;visibility:hidden';
document.body.appendChild(modeProbe);

function resolveToken(name) {
  modeProbe.style.color = 'var(' + name + ')';
  const m = getComputedStyle(modeProbe).color.match(/\d+(\.\d+)?/g);
  if (!m) return null;
  return '#' + m.slice(0, 3).map(c => (+c).toString(16).padStart(2, '0')).join('');
}

/* leitura e escrita do armazenamento sempre em try: em janela privada o
   proprio acessor lança, e uma pagina que quebra por causa da memoria de
   um interruptor e pior que um interruptor sem memoria */
function storedMode() {
  try { return window.localStorage.getItem(MODE_KEY); } catch (e) { return null; }
}
function storeMode(mode) {
  try { window.localStorage.setItem(MODE_KEY, mode); } catch (e) {}
}

/* todo link interno carrega o modo: sem isto a escolha morre no primeiro
   clique da navbar, e e justamente a navbar que leva as outras paginas */
function carryMode(mode) {
  document.querySelectorAll('a[href^="/"]').forEach(function (a) {
    const url = new URL(a.getAttribute('href'), location.origin);
    if (mode === 'dark') url.searchParams.set('mode', 'dark');
    else url.searchParams.delete('mode');
    a.setAttribute('href', url.pathname + (url.search || '') + (url.hash || ''));
  });
}

function applyMode(mode, record) {
  const dark = mode === 'dark';
  document.documentElement.className = dark ? 'dark' : '';
  const btn = document.getElementById('modeBtn');
  if (btn) {
    btn.setAttribute('aria-pressed', String(dark));
    btn.setAttribute('aria-label', dark ? '__mode_to_light__' : '__mode_to_dark__');
  }
  const bg = resolveToken('--bg');
  if (themeMeta && bg) themeMeta.setAttribute('content', bg);
  carryMode(dark ? 'dark' : '');
  if (!record) return;
  storeMode(dark ? 'dark' : 'light');
  const url = new URL(location.href);
  if (dark) url.searchParams.set('mode', 'dark');
  else url.searchParams.delete('mode');
  history.replaceState(null, '', url);
}

(function initMode() {
  const asked = new URL(location.href).searchParams.get('mode');
  const remembered = storedMode();
  const start = asked === 'dark' ? 'dark'
              : asked === 'light' ? 'light'
              : remembered === 'dark' ? 'dark'
              : 'light';
  applyMode(start, false);
  const btn = document.getElementById('modeBtn');
  if (btn) btn.addEventListener('click', function () {
    applyMode(document.documentElement.classList.contains('dark') ? 'light' : 'dark', true);
  });
})();

async function checkServerStatus() {
  const dot = document.getElementById('statusDot');
  const lbl = document.getElementById('statusLabel');
  const t0 = performance.now();
  try {
    const r = await fetch('/health', { cache: 'no-cache' });
    const ms = Math.round(performance.now() - t0);
    if (r.ok) {
      dot.classList.remove('offline'); dot.classList.add('online');
      if (lbl) { lbl.textContent = 'Online (' + ms + 'ms)'; lbl.classList.remove('offline'); lbl.classList.add('online'); }
    } else {
      dot.classList.remove('online'); dot.classList.add('offline');
      if (lbl) { lbl.textContent = 'Offline'; lbl.classList.remove('online'); lbl.classList.add('offline'); }
    }
  } catch(e) {
    dot.classList.remove('online'); dot.classList.add('offline');
    if (lbl) { lbl.textContent = 'Offline'; lbl.classList.remove('online'); lbl.classList.add('offline'); }
  }
}
checkServerStatus();
setInterval(checkServerStatus, 15000);

/* Abrir e fechar com as receitas de pure/motion.css. O display continua
   sendo a chave: a superficie sai do fluxo fechada, entao o teclado nao
   a alcanca. O reflow entre ligar o display e ligar is-open e o que faz
   a transicao correr em vez de saltar. A saida le a duracao do token,
   nunca um numero escrito aqui. */
function motionExitMs() {
  const v = getComputedStyle(document.documentElement).getPropertyValue('--duration-2');
  return parseFloat(v) || 0;
}
function motionOpen(el) {
  el.classList.remove('is-closing');
  el.classList.add('open');
  void el.offsetWidth;
  el.classList.add('is-open');
}
function motionClose(el, done) {
  el.classList.remove('is-open');
  el.classList.add('is-closing');
  window.setTimeout(function() {
    el.classList.remove('open', 'is-closing');
    if (done) done();
  }, motionExitMs());
}

function openFeedback() {
  const overlay = document.getElementById('feedbackOverlay');
  const card = overlay.querySelector('.modal');
  card.classList.remove('is-closing');
  motionOpen(overlay);
  card.classList.add('is-open');
  document.getElementById('feedbackMsg').focus();
}
function closeFeedback() {
  const overlay = document.getElementById('feedbackOverlay');
  const card = overlay.querySelector('.modal');
  card.classList.remove('is-open');
  card.classList.add('is-closing');
  motionClose(overlay, function() {
    card.classList.remove('is-closing');
    document.getElementById('feedbackMsg').value = '';
  });
}
function closeFeedbackOutside(e) {
  if (e.target === document.getElementById('feedbackOverlay')) closeFeedback();
}
function submitFeedback() {
  const type = document.getElementById('feedbackType').value;
  const msg  = document.getElementById('feedbackMsg').value.trim();
  const labels = { bug: 'bug', feature: 'enhancement', other: 'question' };
  const titleMap = { bug: '[Bug] ', feature: '[Feature] ', other: '[Other] ' };
  const title = encodeURIComponent(titleMap[type] + (msg.split('\\n')[0].slice(0, 60) || 'User report'));
  const body  = encodeURIComponent(msg || '(no description provided)');
  const label = encodeURIComponent(labels[type]);
  const url = 'https://github.com/madsondeluna/AMPidentifier/issues/new?title=' + title + '&body=' + body + '&labels=' + label;
  window.open(url, '_blank', 'noopener,noreferrer');
  closeFeedback();
}"""


def page(title, description, path, body, schema='', css='', js='', lang='en'):
    """Monta uma pagina inteira a partir do miolo dela, no idioma pedido."""
    other = 'pt' if lang == 'en' else 'en'
    here = localised(path, lang)
    fills = {k: t(k, lang) for k in STRINGS}
    fills.update({
        'lang_href': localised(path, other),
        'lang_other': 'pt-BR' if other == 'pt' else 'en',
        'lang_code': other.upper(),
    })

    def fill(tpl):
        for k, v in fills.items():
            tpl = tpl.replace('__%s__' % k, v)
        return tpl

    head = HEAD % {'title': title, 'description': description, 'path': here}
    head = head.replace('<html lang="en">',
                        '<html lang="%s">' % ('pt-BR' if lang == 'pt' else 'en'))

    # hreflang reciproco: a anotacao de uma rota so vale se a irma apontar
    # de volta, e x-default fica no ingles, que e a lingua da ferramenta.
    alt = ''.join(
        '<link rel="alternate" hreflang="%s" href="https://www.ampidentifier.com%s">\n' % (h, u)
        for h, u in (('en', localised(path, 'en')),
                     ('pt-BR', localised(path, 'pt')),
                     ('x-default', localised(path, 'en')))
    )
    head = head.replace('<link rel="canonical"', alt + '<link rel="canonical"')

    # a rota corrente se marca no proprio link, e nao por javascript: sem
    # aria-current um leitor de tela nao tem como saber onde esta.
    # o alvo e o data-nav e nao o href, porque href="/" aparece antes no
    # link da marca e a marca nao e a rota corrente
    slug = {'/': 'predict', '/about': 'about', '/suggestions': 'suggestions',
            '/beta': 'beta'}.get(path, '')
    # A reescrita de prefixo corre ANTES do preenchimento, com o href da
    # sigla de idioma ainda como marcador: feita depois, ela prefixava o
    # proprio link de troca e a sigla apontava para a pagina onde ja se
    # esta. Medido: em /pt/about a sigla EN levava a /pt/about.
    nav = NAV
    if slug:
        nav = nav.replace('data-nav="%s"' % slug,
                          'data-nav="%s" aria-current="page"' % slug, 1)
    if lang != 'en':
        nav = nav.replace('href="/beta"', 'href="/pt/beta"')
        nav = nav.replace('href="/about"', 'href="/pt/about"')
        nav = nav.replace('href="/suggestions"', 'href="/pt/suggestions"')
        nav = nav.replace('href="/" data-nav', 'href="/pt" data-nav')
        nav = nav.replace('class="nav-brand" href="/"', 'class="nav-brand" href="/pt"')
    nav = fill(nav)

    return (
        '<!DOCTYPE html>\n' + head + '\n<style>\n' + STYLE + css + '\n</style>\n'
        + schema + '</head>\n<body>\n'
        + DEFS + '\n'
        + fill('<a class="skip-link pill" href="#main">__skip__</a>') + '\n\n'
        + nav + '\n\n<div class="shell">\n' + body + '\n</div>\n\n'
        + fill(FOOTER_BAR) + '\n\n' + fill(MODAL) + '\n\n'
        + '{% raw %}<script>\n' + fill(SHELL_JS) + '\n' + js + '\n</script>{% endraw %}\n'
        + '<script src="/pure/light.js?v={{ asset_v }}" defer></script>\n'
        + '</body>\n</html>'
    )
