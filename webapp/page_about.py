"""About page, served at /beta/about.

Only the body lives here. Head, style, navbar, footer bar and modal come
from page_shell.py and are the same on every route under /beta.
"""

from webapp.page_shell import page

BODY = """
  <header class="step-1">
    <h1 class="page-title">About AMPidentifier</h1>
  </header>

  <main class="step-2" id="main" tabindex="-1">

    <div class="card-glass prose-block prose-justify">
      <p>AMPidentifier classifies peptide sequences as antimicrobial or
      non-antimicrobial from the primary amino acid sequence alone. It runs as a
      web tool, a command line program and a Python package, on the same models
      and the same thresholds in all three.</p>
    </div>

    <div class="metrics-band step-2">
      <div class="metrics-label">Model</div>
      <div class="card-glass prose-block prose-justify">
        <p>The default mode is a soft-voting ensemble of five classifiers: random
        forest, support vector machine, gradient boosting, XGBoost and LightGBM.
        Each sequence is reduced to 22 physicochemical and compositional
        descriptors, among them net charge, hydrophobicity, hydrophobic moment,
        isoelectric point, aliphatic index, instability index and amino acid
        composition.</p>
        <p>On the independent benchmark set the ensemble reaches AUC-ROC 0.950,
        MCC 0.742, sensitivity 94.9% and specificity 78.4%. Single-model modes are
        available and score lower on every one of those four.</p>
      </div>
    </div>

    <div class="metrics-band step-2">
      <div class="metrics-label">Benchmark, voting ensemble</div>
      <div class="metrics-grid metrics-4">
        <div class="card-glass metric"><span class="num metric-val">0.950</span><span class="metric-lbl">AUC-ROC</span></div>
        <div class="card-glass metric"><span class="num metric-val">0.742</span><span class="metric-lbl">MCC</span></div>
        <div class="card-glass metric"><span class="num metric-val">94.9%</span><span class="metric-lbl">Sensitivity</span></div>
        <div class="card-glass metric"><span class="num metric-val">78.4%</span><span class="metric-lbl">Specificity</span></div>
      </div>
    </div>

    <div class="metrics-band step-2">
      <div class="metrics-label">Scope</div>
      <div class="card-glass prose-block prose-justify">
        <p>Training data comes from experimentally validated antimicrobial peptides
        in public databases, against non-antimicrobial sequences. The coverage
        includes antibacterial, antifungal, antiviral and other host defence
        peptides. A protein whose primary function is not antimicrobial can still
        carry antimicrobial features in specific regions of its sequence, so a
        positive call is a hypothesis for testing, not a measurement.</p>
      </div>
    </div>

    <div class="metrics-band step-2">
      <div class="metrics-label">Access</div>
      <div class="card-glass prose-block">
        <p class="install"><span class="install-lead">Python package:</span> <code>pip install ampidentifier</code></p>
        <p class="install"><span class="install-lead">Source and command line:</span> <a href="https://github.com/madsondeluna/AMPidentifier" target="_blank" rel="noopener">github.com/madsondeluna/AMPidentifier</a></p>
        <p class="install"><span class="install-lead">Web:</span> <a href="/">stable version</a> and <a href="/beta">beta layout</a></p>
      </div>
    </div>

    <div class="metrics-band step-2">
      <div class="metrics-label">Privacy</div>
      <div class="card-glass prose-block prose-justify">
        <p>Sequences are processed in memory and are not stored. Transfer is
        encrypted with HTTPS/TLS. No account is required and no sequence data is
        shared with third parties.</p>
      </div>
    </div>

    <div class="metrics-band step-2">
      <div class="metrics-label">Citation</div>
      <div class="card-glass prose-block prose-justify">
        <p>Luna-Arag&atilde;o, M. A., da Silva, R. L., Bezerra Neto, J. P.,
        dos Santos-Silva, C. A., da Silva Santos, D. E. &amp;
        Benko&#8209;Iseppon, A. M. (2026). AMPidentifier: A Cross-Platform
        Ensemble Toolkit for Antimicrobial Peptide Prediction.</p>
        <p>Registered with the <strong lang="pt-BR">INPI &ndash; Instituto Nacional
        da Propriedade Industrial</strong> under No. <strong>BR 51 2025 005859-4</strong>,
        property of the <strong lang="pt-BR">Universidade Federal de Pernambuco
        (UFPE)</strong> and the <strong lang="pt-BR">Laborat&oacute;rio de
        Gen&eacute;tica e Biotecnologia Vegetal (LGBV)</strong>.</p>
      </div>
    </div>

  </main>
"""

CSS = """
  .page-title { font-size: var(--text-32); }
  .prose-block > p + p { margin-top: var(--space-12); }
  .prose-block { padding: var(--space-16) var(--space-24); }
"""

PAGE = page(
    title='About | AMPidentifier',
    description=('How AMPidentifier predicts antimicrobial peptides: the voting '
                 'ensemble, the 22 descriptors, the benchmark figures, the scope '
                 'of the training data and the privacy terms.'),
    path='/beta/about',
    body=BODY,
    css=CSS,
)
