"""Beta front end: the Pure Design page, served at /beta.

The production page lives in app.py and is not touched by this file. Both
render against the same API routes (/predict, /stats, /locations, /health,
/send_csv, /send_recommendation), so only the markup differs. Assets that
diverge from production carry their own path: the ink-cropped logos under
/img/pure/ and the design tokens under /pure/.
"""

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="#ffffff">
<title>AMPidentifier BETA | Antimicrobial Peptide Prediction Tool</title>
<meta name="description" content="AMPidentifier is a free web tool for antimicrobial peptide (AMP) prediction using machine learning ensemble models. Submit FASTA sequences and classify AMPs in seconds.">
<meta name="keywords" content="antimicrobial peptide prediction, antimicrobial peptide predictor, antimicrobial peptide classifier, antimicrobial peptide classification, antimicrobial peptide identification, antimicrobial peptide detection, antimicrobial peptide screening, antimicrobial peptide discovery, antimicrobial peptide annotation, antimicrobial peptide mining, antimicrobial peptide search, antimicrobial activity prediction, peptide bioactivity prediction, peptide function prediction, bioactive peptide prediction, in silico peptide screening, virtual screening peptides, high throughput peptide screening, AMP prediction, AMP predictor, AMP classifier, AMP identification, AMP detection, AMP screening, AMP discovery, AMP annotation, AMP prediction tool, AMP prediction software, AMP prediction server, AMP prediction web server, AMP prediction online, AMP prediction free, AMP prediction API, AMP prediction pipeline, AMP prediction benchmark, AMP prediction accuracy, AMP prediction machine learning, AMP prediction deep learning, AMP finder, AMP scanner, AMP toolkit, machine learning antimicrobial peptides, deep learning antimicrobial peptides, machine learning bioinformatics, deep learning bioinformatics, ensemble learning, ensemble model, soft voting classifier, stacking ensemble, gradient boosting, extreme gradient boosting, XGBoost, LightGBM, random forest, support vector machine, logistic regression, neural network, multilayer perceptron, convolutional neural network, recurrent neural network, BiLSTM, transformer protein model, protein language model, PLLM, embeddings, feature engineering, feature selection, cross validation, hyperparameter tuning, class imbalance, ROC AUC, sensitivity specificity, Matthews correlation coefficient, confusion matrix, model interpretability, SHAP, amino acid composition, AAC, dipeptide composition, DPC, pseudo amino acid composition, CTD descriptors, composition transition distribution, physicochemical descriptors, net charge, hydrophobicity, hydrophobic moment, isoelectric point, aliphatic index, instability index, molecular weight, helical wheel, amphipathicity, peptide length, sequence descriptors, protein descriptors, iFeature, propy, peptides package, bioinformatics tool, bioinformatics web server, computational biology, structural bioinformatics, proteomics, peptidomics, genomics, transcriptomics, metagenomics, immunoinformatics, molecular biology software, sequence analysis, FASTA, FASTA input, multi FASTA, batch prediction, CSV export, command line interface, CLI tool, Python package, pip install ampidentifier, open source bioinformatics, reproducible research, Google Colab notebook, antibiotic resistance, antimicrobial resistance, AMR, multidrug resistant bacteria, superbugs, ESKAPE pathogens, novel antibiotics, antibiotic alternatives, drug discovery, peptide drug design, therapeutic peptides, host defense peptides, innate immunity, defensins, cathelicidins, bacteriocins, lantibiotics, cecropins, magainins, LL-37, antibacterial peptides, antifungal peptides, antiviral peptides, antiparasitic peptides, antibiofilm peptides, anticancer peptides, cell penetrating peptides, hemolytic activity, cytotoxicity prediction, minimum inhibitory concentration, MIC prediction, plant antimicrobial peptides, insect antimicrobial peptides, marine antimicrobial peptides, APD3, DBAASP, DRAMP, CAMPR3, LAMP database, UniProt, SwissProt, AMP database, curated peptide dataset, training dataset, benchmark dataset, AMPidentifier, ampidentifier, AMPidentifier web, AMPidentifier CLI, AMPidentifier Python, free AMP prediction tool, online AMP predictor, no login bioinformatics tool, AMP prediction 2026, new AMP prediction tool, predição de peptídeos antimicrobianos, peptídeo antimicrobiano, identificação de peptídeos antimicrobianos, classificador de peptídeos antimicrobianos, ferramenta de bioinformática, aprendizado de máquina, aprendizado profundo, resistência antimicrobiana, resistência a antibióticos, descoberta de fármacos, análise de sequências, peptídeos bioativos, peptídeos de defesa, ferramenta gratuita online, 抗菌肽预测, 抗菌肽, 抗菌肽识别, 抗菌肽分类器, 生物信息学工具, 机器学习, 深度学习, 抗生素耐药性, 药物发现, 序列分析, 在线预测工具, 免费工具, रोगाणुरोधी पेप्टाइड भविष्यवाणी, रोगाणुरोधी पेप्टाइड, पेप्टाइड वर्गीकरण, जैव सूचना विज्ञान उपकरण, मशीन लर्निंग, डीप लर्निंग, एंटीबायोटिक प्रतिरोध, दवा खोज, अनुक्रम विश्लेषण, मुफ्त ऑनलाइन उपकरण">
<meta name="author" content="Madson Aragao">
<meta name="robots" content="index, follow">
<link rel="canonical" href="https://www.ampidentifier.com/beta">
<meta property="og:locale" content="en_US">
<meta property="og:locale:alternate" content="pt_BR">
<meta property="og:locale:alternate" content="zh_CN">
<meta property="og:locale:alternate" content="hi_IN">
<meta property="og:type" content="website">
<meta property="og:url" content="https://www.ampidentifier.com/beta">
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@125,300..400&family=Public+Sans:wght@300;400;500;600&family=Spline+Sans+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/pure/tokens.css?v={{ asset_v }}">
<link rel="stylesheet" href="/pure/patterns.css?v={{ asset_v }}">
<style>
  /* =========================================================
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
  .prose-justify { -webkit-hyphens: auto; hyphens: auto; }

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
  .logo-row { display: flex; align-items: center; flex-wrap: nowrap; gap: var(--space-6); }
  /* altura igual nao e tamanho igual. Os arquivos ja vem cortados na
     propria tinta, entao o que sobra de diferenca e a forma da marca:
     um logotipo de uma linha gasta a altura toda numa fileira de letras,
     um empilhado divide a mesma altura entre marca e legenda e sai com
     metade do tamanho aparente. Quem tem proporcao abaixo de 1.6 sobe
     um degrau. */
  .logo-row img {
    display: block;
    height: var(--space-32);
    width: auto;
    object-fit: contain;
    filter: grayscale(1);
    opacity: 0.6;
    transition: opacity var(--duration-3) var(--ease-standard);
  }
  .logo-row img.logo-stacked { height: var(--space-40); }
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
</style>
</head>
<body>
<a class="skip-link pill" href="#main">Skip to content</a>

<div class="shell">

  <header>
    <div>
      <!-- a marca e a imagem; o titulo da pagina existe para leitor de tela
           e para a hierarquia do documento, que nao tinha nenhum h1 -->
      <h1 class="sr-only">AMPidentifier, antimicrobial peptide prediction</h1>
      <img src="/img/logo.svg" alt="AMPidentifier" class="brand-logo">
      <div class="status-row">
        <span class="tip status-tip" tabindex="0" aria-describedby="statusTip">
          <span class="status-dot" id="statusDot"></span>
          <span class="status-label" id="statusLabel">Checking</span>
          <span id="statusTip" role="tooltip">
            <span class="tt-row"><span class="tt-dot c-good"></span> Online: model loaded, predictions ready</span>
            <span class="tt-row"><span class="tt-dot c-crit"></span> Offline: backend unreachable, try again shortly</span>
            <span class="tt-row"><span class="tt-dot c-idle"></span> Checking server status</span>
            <span class="tt-row tt-note">(Xms) = current /health round-trip latency</span>
          </span>
        </span>
      </div>
    </div>

    <div class="metrics-band step-1">
     <div class="card-glass intro">
      <p class="sub prose-justify"><strong>AMPidentifier</strong> is a toolkit for antimicrobial peptide prediction using ensemble machine learning.</p>

      <div class="install-stack">
        <p class="install"><span class="install-lead">For <a href="https://pypi.org/project/ampidentifier/" target="_blank">PyPI</a>:</span> <code>pip install ampidentifier</code></p>
        <p class="install"><span class="install-lead">For terminal use:</span> <a href="https://github.com/madsondeluna/AMPIdentifier" target="_blank">CLI version</a></p>
        <p class="install"><span class="install-lead">This is the beta layout:</span> <a href="/">Access the stable version</a></p>
      </div>
     </div>
    </div>
  </header>

  <div class="metrics-band step-2">
    <div class="metrics-label">In testing</div>
    <div class="card-glass changelog">
      <p>This round changes the interface only. Models, thresholds and predictions are the same as the stable version.</p>
      <p class="changelog-body prose-justify">The front end was rebuilt on a token-based design system: one type scale, one spacing scale and a single set of colour tokens shared by every component, with the layout on a single column and a concentric radius ladder. Controls and panels became glass surfaces with backdrop-filter, keyboard focus rings and reduced-motion fallbacks, the usage map became inline SVG instead of a tile layer, and the result panel carries its state in the URL.</p>
      <p class="changelog-body prose-justify">Coming soon: a new batch of trained models will reach the beta before the stable version, and a prediction mode built on a protein language model (PLLM) goes into testing here.</p>
    </div>
  </div>

  <div class="metrics-band step-2">
    <div class="metrics-label">Benchmark, voting ensemble (RF + SVM + GB + XGB + LGBM)</div>
    <div class="metrics-grid metrics-4">
      <div class="card-glass metric"><span class="num metric-val">0.950</span><span class="metric-lbl">AUC-ROC</span></div>
      <div class="card-glass metric"><span class="num metric-val">0.742</span><span class="metric-lbl">MCC</span></div>
      <div class="card-glass metric"><span class="num metric-val">94.9%</span><span class="metric-lbl">Sensitivity</span></div>
      <div class="card-glass metric"><span class="num metric-val">78.4%</span><span class="metric-lbl">Specificity</span></div>
    </div>
  </div>

  <div class="metrics-band step-1">
    <div class="metrics-label">Usage</div>
    <div class="metrics-grid metrics-4">
      <div class="card-glass metric"><span class="num metric-val" id="statSeq">&mdash;</span><span class="metric-lbl">Sequences classified</span></div>
      <div class="card-glass metric"><span class="num metric-val" id="statVisitors">&mdash;</span><span class="metric-lbl">Unique users</span></div>
      <div class="card-glass metric"><span class="num metric-val" id="statRuns">&mdash;</span><span class="metric-lbl">Prediction runs</span></div>
      <div class="card-glass metric"><span class="num metric-val">22</span><span class="metric-lbl">Descriptors</span></div>
    </div>
  </div>

  <main class="step-2" id="main" tabindex="-1">

    <div class="surface share-section step-2">
      <div class="share-inner">
        <div class="share-heading">Find AMPidentifier useful?</div>
        <div class="share-actions">
          <button class="pill" onclick="copyLink()" id="copyLinkBtn">Copy link</button>
          <button class="pill" onclick="toggleShareForm()" id="shareEmailBtn">Share by email</button>
        </div>
      </div>
      <div class="share-url-box mono" id="shareUrlBox"></div>
      <div class="share-form" id="shareForm">
        <span class="select-shell">
          <select class="select" id="shareLang" title="Email language">
            <option value="en">English</option>
            <option value="fr">Français</option>
            <option value="es">Español</option>
            <option value="pt">Português</option>
            <option value="zh">中文</option>
          </select>
        </span>
        <label class="field">
          <span class="sr-only">Recipient email</span>
          <input class="input" type="email" id="shareFriendEmail" placeholder="friend@example.com" autocomplete="email" spellcheck="false">
        </label>
        <button class="pill" onclick="sendShareEmail()" id="sendShareBtn">Send</button>
        <div class="share-form-status" id="shareFormStatus" aria-live="polite"></div>
      </div>
    </div>

    <div class="step-2">
      <div class="label-row">
        <label class="field-label" for="fasta">FASTA sequences</label>
        <span class="seq-counter" id="seqCounter"></span>
      </div>
      <textarea class="textarea" id="fasta" spellcheck="false" placeholder=">SequenceID
KRIVQRIKDFLRNLVPRTES" oninput="updateCounter();validateFasta();"></textarea>
      <div id="validationErr" aria-live="polite"></div>

      <input type="file" id="fileInput" accept=".fasta,.fa,.txt" onchange="handleFileUpload(event)">

      <div class="row">
        <span class="select-shell">
          <select class="select" id="model" aria-label="Model">
            <option value="voting">Voting ensemble</option>
            <option value="rf">Random forest</option>
            <option value="svm">SVM</option>
            <option value="gb">Gradient boosting</option>
            <option value="xgb">XGBoost</option>
            <option value="lgbm">LightGBM</option>
          </select>
        </span>
        <button class="pill glass-accent" id="runBtn" onclick="runPrediction()">Run</button>
        <button class="pill" onclick="clearAll()">Clear</button>
        <button class="pill" onclick="loadExample()">Load example</button>
        <button class="pill" onclick="document.getElementById('fileInput').click()">Upload .fasta</button>
      </div>

      <div id="status" aria-live="polite"></div>
    </div>

    <div id="results"></div>
  </main>

  <footer class="step-2">
    <div>
      <div class="usage-map-title">Where AMPidentifier is being used</div>
      <div id="usageMap"></div>
    </div>

    <div class="step-2 prose-justify">
      <p>Luna-Aragão, M. A., da Silva, R. L., Bezerra Neto, J. P., dos Santos-Silva, C. A., da Silva Santos, D. E. &amp; Benko&#8209;Iseppon, A. M. (2026).
      AMPidentifier: A Cross-Platform Ensemble Toolkit for Antimicrobial Peptide Prediction.
      GitHub repository: <a href="https://github.com/madsondeluna/AMPIdentifier" target="_blank">https://github.com/madsondeluna/AMPIdentifier</a></p>
      <!-- a pagina e em ingles e os nomes proprios sao em portugues: sem o
           lang, o navegador tenta hifenizar "Universidade" e "Biotecnologia"
           pelo dicionario errado, desiste, e a linha justificada estica o
           espaco entre as palavras para fechar a margem -->
      <p>This tool is officially registered with the <strong lang="pt-BR">INPI &ndash; Instituto Nacional da Propriedade Industrial</strong> (Brazilian National Institute of Industrial Property), Registration No. <strong>BR 51 2025 005859-4</strong>. It is a property of the <strong lang="pt-BR">Universidade Federal de Pernambuco (UFPE)</strong> and the <strong lang="pt-BR">Laboratório de Genética e Biotecnologia Vegetal (LGBV)</strong>.</p>
      <p>Your data is encrypted during transfer (HTTPS/TLS) and never shared. Sequences are not stored.</p>
      <p>Developer: <a href="mailto:madsondeluna@gmail.com">madsondeluna@gmail.com</a> &nbsp;·&nbsp; <a href="https://madsondeluna.com" target="_blank">madsondeluna.com</a> &nbsp;·&nbsp; <button class="feedback-link" onclick="openFeedback()">Report issue or suggestion</button> &nbsp;·&nbsp; <span class="version">v{{ version }}</span></p>
    </div>
    <!-- a categoria de cada marca sai do alt, que continua completo: o
         rotulo visivel repetia o que a imagem ja diz e cobrava altura -->
    <div class="logo-strip step-1">
      <div class="logo-group">
        <div class="logo-group-label">Institutions</div>
        <div class="logo-row">
          <img src="/img/pure/ufpe.png"     alt="Universidade Federal de Pernambuco">
          <img src="/img/pure/ufmg.png"     alt="Universidade Federal de Minas Gerais">
          <img src="/img/pure/upe-logo.png" alt="Universidade de Pernambuco" class="logo-stacked">
        </div>
      </div>
      <div class="logo-group">
        <div class="logo-group-label">Departments</div>
        <div class="logo-row">
          <img src="/img/pure/dqf.png"   alt="Departamento de Química Fundamental, UFPE" class="logo-stacked">
          <img src="/img/pure/dgen.jpeg" alt="Departamento de Genética, UFPE"            class="logo-stacked">
        </div>
      </div>
      <div class="logo-group">
        <div class="logo-group-label">Funding</div>
        <div class="logo-row">
          <img src="/img/pure/facepe.png"  alt="FACEPE">
          <img src="/img/pure/fapemig.png" alt="FAPEMIG" class="logo-stacked">
        </div>
      </div>
      <div class="logo-group">
        <div class="logo-group-label">Research groups</div>
        <div class="logo-row">
          <img src="/img/pure/lgbv.png" alt="Laboratório de Genética e Biotecnologia Vegetal">
          <img src="/img/pure/lcm3.png" alt="LCM3">
        </div>
      </div>
    </div>

  </footer>

</div>

<!-- Feedback modal -->
<div class="modal-overlay" id="feedbackOverlay" onclick="closeFeedbackOutside(event)">
  <div class="modal modal-card surface" role="dialog" aria-modal="true" aria-labelledby="feedbackTitle">
    <h2 id="feedbackTitle">Report issue or suggestion</h2>
    <div class="field">
      <label class="field-label" for="feedbackType">Type</label>
      <span class="select-shell">
        <select class="select" id="feedbackType">
          <option value="bug">Bug report</option>
          <option value="feature">Feature request</option>
          <option value="other">Other</option>
        </select>
      </span>
    </div>
    <div class="field">
      <label class="field-label" for="feedbackMsg">Description</label>
      <textarea class="textarea" id="feedbackMsg" placeholder="Describe the issue or your suggestion..."></textarea>
    </div>
    <div class="modal-actions">
      <button class="pill" onclick="closeFeedback()">Cancel</button>
      <button class="pill glass-accent" onclick="submitFeedback()">Open on GitHub</button>
    </div>
  </div>
</div>

{% raw %}<script>
const EXAMPLE = [
  ">Magainin-2|Xenopus_laevis|Cationic_amphipathic_helix",
  "GIGKFLHSAKKFGKAFVGEIMNS",
  ">LL-37|Homo_sapiens|Cathelicidin_family",
  "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
  ">Melittin|Apis_mellifera|Venom_peptide",
  "GIGAVLKVLTTGLPALISWIKRKRQQ",
  ">Insulin_Chain_B|Homo_sapiens|Peptide_hormone",
  "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
  ">Glucagon|Homo_sapiens|Peptide_hormone",
  "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT",
  ">Vasoactive_intestinal_peptide|Homo_sapiens|Neuropeptide",
  "HSDAVFTDNYTRLRKQMAVKKYLNSILN"
].join("\\n");

const VALID_AA = /^[ACDEFGHIKLMNPQRSTVWYBXZUOJ*-]+$/i;
let lastData = null;
let lastModel = null;

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

async function loadStats() {
  try {
    const r = await fetch('/stats', { cache: 'no-cache' });
    const d = await r.json();
    const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v != null ? v.toLocaleString() : '—'; };
    set('statSeq',      d.total_sequences);
    set('statRuns',     d.total_runs);
    set('statVisitors', d.unique_sessions);
  } catch(e) {}
}
loadStats();

function initUsageMap() {
  const host = document.getElementById('usageMap');
  if (!host) return;
  const NS = 'http://www.w3.org/2000/svg';
  const el = function(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };

  /* vazio, cheio e erro sao tres desenhos aqui tambem: o mapa sem dado
     nenhum nao pode ser o mesmo mundo cinza do mapa que falhou. */
  const showNote = function(text) {
    host.classList.add('is-note');
    host.innerHTML = '';
    const box = document.createElement('div');
    box.className = 'empty';
    box.textContent = text;
    host.appendChild(box);
  };

  Promise.all([
    fetch('/map-outline.json?v=1').then(function(r) { return r.json(); }),
    fetch('/locations', { cache: 'no-cache' }).then(function(r) { return r.json(); }),
  ]).then(function(res) {
    const world = res[0];
    let rows = res[1];
    if (!world || !Array.isArray(rows)) { showNote('Usage map unavailable.'); return; }
    rows = rows.filter(function(d) { return d.lat != null && d.lon != null; });
    if (!rows.length) { showNote('No usage recorded yet.'); return; }

    const scale = world.w / 360, latTop = 84;
    const px = function(lon) { return (lon + 180) * scale; };
    const py = function(lat) { return (latTop - lat) * scale; };

    const svg = el('svg', {
      viewBox: '0 0 ' + world.w + ' ' + world.h,
      role: 'img',
      'aria-label': 'World map of AMPidentifier usage',
    }, host);

    /* vidro num <circle> nao existe: backdrop-filter nao se aplica a
       elemento SVG em nenhum motor. O que faz a bolha ler como vidro sem
       o desfoque e a geometria: corpo translucido, aro claro e um brilho
       especular no alto. O brilho e um gradiente radial deslocado, e sai
       do mesmo --glass-specular que a linguagem ja usa nos controles. */
    const defs = el('defs', {}, svg);
    const gloss = el('radialGradient', { id: 'ringGloss', cx: '0.35', cy: '0.28', r: '0.72' }, defs);
    el('stop', { offset: '0',    class: 'gloss-in'  }, gloss);
    el('stop', { offset: '0.62', class: 'gloss-mid' }, gloss);
    el('stop', { offset: '1',    class: 'gloss-out' }, gloss);

    el('path', { d: world.d, class: 'land' }, svg);

    const tip = document.createElement('div');
    tip.className = 'map-tip';
    tip.innerHTML = '<div class="place"></div><div class="value"></div>';
    host.appendChild(tip);

    const showTip = function(target, place, value) {
      tip.querySelector('.place').textContent = place;
      tip.querySelector('.value').textContent = value;
      tip.style.opacity = '1';
      const hb = host.getBoundingClientRect();
      const tb = target.getBoundingClientRect();
      const half = tip.offsetWidth / 2;
      let left = tb.left - hb.left + tb.width / 2;
      left = Math.max(half + 2, Math.min(hb.width - half - 2, left));
      tip.style.left = left + 'px';
      tip.style.top  = Math.max(tip.offsetHeight + 2, tb.top - hb.top - 8) + 'px';
    };
    const hideTip = function() { tip.style.opacity = '0'; };
    document.addEventListener('click', function(ev) {
      if (!ev.target.closest || !ev.target.closest('.spot')) hideTip();
    });

    const total = rows.reduce(function(s, d) { return s + (d.count || 0); }, 0);
    const max = rows.reduce(function(m, d) { return Math.max(m, d.count || 0); }, 1);
    const rings = [];

    // rings scale with the map, with a floor in screen pixels so the smallest
    // ones stay visible when the SVG shrinks on narrow viewports
    const sizeRings = function() {
      const unit = (svg.getBoundingClientRect().width || world.w) / world.w;
      if (!unit) return;
      const floor = 4.5 / unit;
      rings.forEach(function(item) {
        const r = Math.max(item.r, floor);
        item.node.setAttribute('r', r);
        item.gloss.setAttribute('r', r);
      });
    };

    rows.slice().sort(function(a, b) { return a.count - b.count; }).forEach(function(d) {
      const x = px(d.lon), y = py(d.lat);
      const r = 5.5 + 19 * Math.sqrt(d.count / max);
      const place = d.city ? (d.city + ', ' + d.country) : (d.country || 'Unknown');
      const p = total > 0 ? (d.count / total * 100) : 0;
      const pct = p < 1 ? '<1%' : Math.round(p) + '%';
      const value = pct + ' of all predictions';
      const g = el('g', { class: 'spot', tabindex: '0', role: 'img',
                          'aria-label': place + ', ' + value }, svg);
      const ring = el('circle', { cx: x, cy: y, r: r, class: 'ring' }, g);
      const lit  = el('circle', { cx: x, cy: y, r: r, class: 'gloss' }, g);
      rings.push({ node: ring, gloss: lit, r: r });
      const show = function() { showTip(ring, place, value); };
      g.addEventListener('mouseenter', show);
      g.addEventListener('focus',      show);
      g.addEventListener('click',      show);
      g.addEventListener('mouseleave', hideTip);
      g.addEventListener('blur',       hideTip);
    });

    sizeRings();
    let resizeTimer = null;
    window.addEventListener('resize', function() {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function() { hideTip(); sizeRings(); }, 150);
    });
  }).catch(function() { showNote('Usage map unavailable.'); });
}
initUsageMap();

function updateCounter() {
  const n = (document.getElementById('fasta').value.match(/^>/gm) || []).length;
  document.getElementById('seqCounter').textContent =
    n > 0 ? n + ' sequence' + (n === 1 ? '' : 's') : '';
}

function setValidationError(msg) {
  const el = document.getElementById('validationErr');
  el.textContent = msg;
  el.className = msg ? 'field-error' : '';
}

function validateFasta() {
  const text  = document.getElementById('fasta').value.trim();
  if (!text) { setValidationError(''); return true; }

  const lines = text.split('\\n').map(l => l.trim()).filter(Boolean);
  if (!lines[0].startsWith('>')) {
    setValidationError('Invalid format: first line must start with >.');
    return false;
  }

  let seq = '', headers = 0;
  for (const line of lines) {
    if (line.startsWith('>')) {
      if (seq) {
        if (seq.length < 5) { setValidationError('Sequence too short (min 5 residues).'); return false; }
        if (!VALID_AA.test(seq)) { setValidationError('Invalid characters in sequence.'); return false; }
      }
      seq = ''; headers++;
    } else {
      seq += line;
    }
  }
  if (seq) {
    if (seq.length < 5) { setValidationError('Sequence too short (min 5 residues).'); return false; }
    if (!VALID_AA.test(seq)) { setValidationError('Invalid characters in sequence.'); return false; }
  }
  if (!headers) { setValidationError('No valid FASTA sequences found.'); return false; }
  setValidationError('');
  return true;
}

function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById('fasta').value = ev.target.result;
    updateCounter();
    setValidationError('');
  };
  reader.readAsText(file);
}

function loadExample() {
  document.getElementById('fasta').value = EXAMPLE;
  updateCounter();
  setValidationError('');
}

function clearAll() {
  document.getElementById('fasta').value = '';
  document.getElementById('status').textContent = '';
  document.getElementById('seqCounter').textContent = '';
  setValidationError('');
  document.getElementById('fileInput').value = '';
  lastData = null;
  lastModel = null;
  showEmptyResults();
}

/* vazio, carregando, cheio e erro sao quatro desenhos. O vazio oferece o
   proximo passo, mas nao repete um botao: "Load example" ja esta na fila
   de acoes logo acima, visivel ao mesmo tempo que este aviso. */
function showEmptyResults() {
  document.getElementById('results').innerHTML =
    '<div class="empty">' +
      '<div><div class="empty-head">No predictions yet</div>' +
      '<div>Paste FASTA sequences above, or start from the example.</div></div>' +
    '</div>';
}

function showResultsSkeleton() {
  const stat = '<div><div class="skeleton sk-val"></div><div class="skeleton skeleton-line"></div></div>';
  const row  = '<div class="skeleton sk-row"></div>';
  document.getElementById('results').innerHTML =
    '<div class="summary surface">' +
      '<div class="skeleton skeleton-line"></div>' +
      '<div class="sk-grid">' + stat + stat + stat + '</div>' +
    '</div>' +
    '<div class="sk-rows">' + row + row + row + row + '</div>';
}

async function runPrediction() {
  const fasta  = document.getElementById('fasta').value.trim();
  const model  = document.getElementById('model').value;
  const btn    = document.getElementById('runBtn');
  const status = document.getElementById('status');
  const field  = document.getElementById('fasta');

  /* formulario incompleto nao trava o envio: o erro aparece no campo e o
     foco vai para ele, em vez de sair num aviso solto abaixo do bloco. */
  if (!fasta) {
    setValidationError('Paste at least one FASTA sequence.');
    field.focus();
    return;
  }
  if (!validateFasta()) { field.focus(); return; }

  btn.disabled = true;
  status.textContent = 'Running prediction...';
  showResultsSkeleton();

  const form = new FormData();
  form.append('fasta_sequence', fasta);
  form.append('model', model);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: form });
    const data = await res.json();
    if (data.error) {
      status.innerHTML = '<span class="err">Error: ' + data.error + '</span>';
      showEmptyResults();
    } else {
      lastData = data.predictions;
      lastModel = data.model;
      status.textContent = '';
      renderResults(data);
      loadStats();
    }
  } catch (e) {
    status.innerHTML = '<span class="err">Request failed: ' + e.message + '</span>';
    showEmptyResults();
  } finally {
    btn.disabled = false;
  }
}

function renderResults(data) {
  const preds  = data.predictions;
  const ampKey  = 'prediction';
  const probKey = 'probability_AMP';

  const amps  = preds.filter(r => r[ampKey] === 1).length;
  const total = preds.length;

  function makeRow(r) {
    const isAmp  = r[ampKey] === 1;
    const prob   = r[probKey] != null ? r[probKey] : null;
    const pct    = prob !== null ? (prob * 100).toFixed(1) + '%' : '—';
    const series = isAmp ? 'is-amp' : 'is-non';
    const fill   = prob !== null ? prob.toFixed(3) : '0';
    const barHtml = prob !== null
      ? '<span class="prob-bar"><span class="prob-fill ' + series + '" style="--fill:' + fill + '"></span></span><span class="num prob-text">' + pct + '</span>'
      : '—';
    const label =
      '<span class="pred"><span class="pred-dot ' + series + '"></span>' +
      (isAmp ? 'AMP' : 'non-AMP') + '</span>';
    return '<tr class="' + (isAmp ? 'r-amp' : 'r-non') + '"><td class="seq-id">' +
      (r.ID || r.id || '—') + '</td><td class="seq-id">' +
      (r.sequence || '—') + '</td><td>' +
      label + '</td><td class="prob-cell">' +
      barHtml + '</td></tr>';
  }

  const modelLabels = {
    voting: 'Voting ensemble',
    rf: 'Random forest',
    svm: 'SVM',
    gb: 'Gradient boosting',
    xgb: 'XGBoost',
    lgbm: 'LightGBM'
  };

  document.getElementById('results').innerHTML =
    '<div class="summary surface">' +
      '<div class="summary-title">Results, ' + (modelLabels[data.model] || data.model) + '</div>' +
      '<div class="summary-grid">' +
        '<div class="stat"><div class="num stat-val">' + total + '</div><div class="stat-label">Sequences</div></div>' +
        '<div class="stat"><div class="num stat-val">' + amps + '</div><div class="stat-label">Predicted AMP</div></div>' +
        '<div class="stat"><div class="num stat-val">' + (total - amps) + '</div><div class="stat-label">Predicted non-AMP</div></div>' +
      '</div>' +
    '</div>' +
    '<div class="filter-row">' +
      '<button class="pill pill-sm hit filter-btn" id="fAll" aria-pressed="true" onclick="applyFilter(\\'all\\')">All</button>' +
      '<button class="pill pill-sm hit filter-btn" id="fAmp" aria-pressed="false" onclick="applyFilter(\\'amp\\')">AMP only</button>' +
      '<button class="pill pill-sm hit filter-btn" id="fNon" aria-pressed="false" onclick="applyFilter(\\'non\\')">Non-AMP only</button>' +
    '</div>' +
    '<div class="table-scroll">' +
      '<table id="tbl">' +
        '<thead><tr><th>ID</th><th>Sequence</th><th>Prediction</th><th>Prob. AMP</th></tr></thead>' +
        '<tbody>' + preds.map(makeRow).join('') + '</tbody>' +
      '</table>' +
    '</div>' +
    '<div class="dl">' +
      '<button class="pill" onclick="downloadCSV()">Download CSV</button>' +
      '<button class="pill" id="copyBtn" onclick="copyTable()">Copy table</button>' +
    '</div>' +
    '<div class="email-csv-section surface">' +
      '<div class="email-csv-title">Receive results by email</div>' +
      '<div class="email-csv-fields">' +
        '<div class="field">' +
          '<label class="field-label" for="csvEmailLang">Language</label>' +
          '<span class="select-shell">' +
            '<select class="select" id="csvEmailLang">' +
              '<option value="en">English</option>' +
              '<option value="fr">Français</option>' +
              '<option value="es">Español</option>' +
              '<option value="pt">Português</option>' +
              '<option value="zh">中文</option>' +
            '</select>' +
          '</span>' +
        '</div>' +
        '<div class="field">' +
          '<label class="field-label" for="csvEmail">Your email</label>' +
          '<input class="input" type="email" id="csvEmail" placeholder="you@example.com" autocomplete="email" spellcheck="false">' +
        '</div>' +
        '<button class="pill" id="sendCsvBtn" onclick="sendCsvByEmail()">Send</button>' +
      '</div>' +
      '<div class="email-csv-status" id="emailCsvStatus" aria-live="polite"></div>' +
    '</div>' +
    '<div class="result-note prose-justify">' +
      '<strong>Interpretation note:</strong> Predictions are computed from 22 physicochemical and compositional descriptors derived from the primary amino acid sequence. ' +
      'For higher predictive power, use the <strong>voting ensemble</strong> mode (RF + SVM + GB + XGB + LGBM), which combines five independent classifiers by soft voting and achieves ' +
      '<strong>AUC-ROC 0.950</strong>, <strong>MCC 0.742</strong>, <strong>Sensitivity 94.9%</strong>, and <strong>Specificity 78.4%</strong> on the independent benchmark set. ' +
      'Bear in mind that proteins whose primary function is not antimicrobial activity may still harbour potential antimicrobial features in specific sequence regions. A full benchmark is available in Luna-Arago et al. (2026), <a href="https://doi.org/10.1021/acs.jcim.XXXXXXX" target="_blank">doi:10.1021/acs.jcim.XXXXXXX</a>.' +
    '</div>';

  applyFilter(currentFilter);
}

function applyFilter(type) {
  document.querySelectorAll('.filter-btn').forEach(b => b.setAttribute('aria-pressed', 'false'));
  const ids = { all: 'fAll', amp: 'fAmp', non: 'fNon' };
  document.getElementById(ids[type]).setAttribute('aria-pressed', 'true');
  document.querySelectorAll('#tbl tbody tr').forEach(row => {
    if (type === 'all') row.style.display = '';
    else if (type === 'amp') row.style.display = row.classList.contains('r-amp') ? '' : 'none';
    else row.style.display = row.classList.contains('r-non') ? '' : 'none';
  });
  currentFilter = type;
  syncUrl();
}

/* modelo e filtro sao estado: quem recarrega a pagina ou manda o link
   adiante chega no mesmo lugar de onde saiu. */
let currentFilter = 'all';

function syncUrl() {
  const p = new URLSearchParams();
  const model = document.getElementById('model').value;
  if (model !== 'voting') p.set('model', model);
  if (currentFilter !== 'all') p.set('filter', currentFilter);
  const q = p.toString();
  history.replaceState(null, '', window.location.pathname + (q ? '?' + q : ''));
}

function restoreFromUrl() {
  const p = new URLSearchParams(window.location.search);
  const model = p.get('model');
  const select = document.getElementById('model');
  if (model && Array.from(select.options).some(o => o.value === model)) select.value = model;
  const filter = p.get('filter');
  if (filter === 'amp' || filter === 'non') currentFilter = filter;
  select.addEventListener('change', syncUrl);
}
restoreFromUrl();
showEmptyResults();

function downloadCSV() {
  if (!lastData) return;
  const keys = Object.keys(lastData[0]);
  const csv  = [
    keys.join(','),
    ...lastData.map(r => keys.map(k => JSON.stringify(r[k] ?? '')).join(','))
  ].join('\\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'ampidentifier_results.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(a.href);
}

function copyTable() {
  if (!lastData) return;
  const keys = Object.keys(lastData[0]);
  const tsv  = [
    keys.join('\\t'),
    ...lastData.map(r => keys.map(k => r[k] ?? '').join('\\t'))
  ].join('\\n');
  navigator.clipboard.writeText(tsv).then(() => {
    const btn = document.getElementById('copyBtn');
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy table', 1500);
  }).catch(() => alert('Copy not supported in this browser.'));
}

async function sendCsvByEmail() {
  if (!lastData || !lastModel) return;
  const email = document.getElementById('csvEmail').value.trim();
  const lang  = document.getElementById('csvEmailLang').value;
  const status = document.getElementById('emailCsvStatus');
  const btn = document.getElementById('sendCsvBtn');
  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    status.innerHTML = '<span class="err">Enter a valid email address.</span>';
    return;
  }
  const keys = Object.keys(lastData[0]);
  const csv = [
    keys.join(','),
    ...lastData.map(r => keys.map(k => JSON.stringify(r[k] ?? '')).join(','))
  ].join('\\n');
  const total = lastData.length;
  const amps  = lastData.filter(r => r.prediction === 1).length;
  btn.disabled = true;
  status.classList.remove('status-good');
  status.textContent = 'Sending...';
  try {
    const form = new FormData();
    form.append('to_email', email);
    form.append('csv_data', csv);
    form.append('lang',  lang);
    form.append('model', lastModel);
    form.append('total', total);
    form.append('amps',  amps);
    const res = await fetch('/send_csv', { method: 'POST', body: form });
    const data = await res.json();
    if (data.ok) {
      status.classList.add('status-good');
      status.textContent = 'Email sent to ' + email;
    } else {
      status.innerHTML = '<span class="err">' + (data.error || 'Failed to send.') + '</span>';
    }
  } catch (e) {
    status.innerHTML = '<span class="err">Request failed: ' + e.message + '</span>';
  } finally {
    btn.disabled = false;
  }
}

function copyLink() {
  const url = window.location.origin + '/';
  const box = document.getElementById('shareUrlBox');
  const btn = document.getElementById('copyLinkBtn');
  navigator.clipboard.writeText(url).then(() => {
    box.textContent = url + '  (copied)';
    box.style.display = 'block';
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy link'; box.style.display = 'none'; }, 2500);
  }).catch(() => {
    box.textContent = url;
    box.style.display = 'block';
  });
}

function toggleShareForm() {
  const form = document.getElementById('shareForm');
  const opening = !form.classList.contains('open');
  form.classList.toggle('open');
  if (opening) document.getElementById('shareFriendEmail').focus();
}

async function sendShareEmail() {
  const email = document.getElementById('shareFriendEmail').value.trim();
  const status = document.getElementById('shareFormStatus');
  const btn = document.getElementById('sendShareBtn');
  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    status.innerHTML = '<span class="err">Enter a valid email.</span>';
    return;
  }
  btn.disabled = true;
  status.classList.remove('status-good');
  status.textContent = 'Sending...';
  try {
    const fd = new FormData();
    fd.append('to_email', email);
    fd.append('lang', document.getElementById('shareLang').value);
    const res = await fetch('/send_recommendation', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.ok) {
      status.classList.add('status-good');
      status.textContent = 'Recommendation sent to ' + email;
      document.getElementById('shareFriendEmail').value = '';
    } else {
      status.innerHTML = '<span class="err">' + (data.error || 'Failed to send.') + '</span>';
    }
  } catch (e) {
    status.innerHTML = '<span class="err">' + e.message + '</span>';
  } finally {
    btn.disabled = false;
  }
}

function openFeedback() {
  document.getElementById('feedbackOverlay').classList.add('open');
  document.getElementById('feedbackMsg').focus();
}
function closeFeedback() {
  document.getElementById('feedbackOverlay').classList.remove('open');
  document.getElementById('feedbackMsg').value = '';
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
}

</script>{% endraw %}
</body>
</html>"""

