"""About page, served at /about.

Only the body lives here. Head, style, navbar, footer bar and modal come
from page_shell.py and are the same on every route under /beta.
"""

from pathlib import Path

from webapp.page_shell import page

# A foto entra so quando o arquivo existe: um <img> apontando para 404
# rende o icone de imagem quebrada, que e pior que nao ter retrato.
_PHOTO = Path(__file__).with_name('img') / 'madson.jpg'
PHOTO_TAG = ('<img class="lead-photo" src="/img/madson.jpg" '
             'alt="Madson Allan de Luna Arag&atilde;o">') if _PHOTO.exists() else ''

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
      <div class="metrics-label">Manifesto</div>
      <div class="card-glass prose-block prose-justify">
        <p>Antimicrobial resistance kills people now and will kill more. The
        sequence space where new antimicrobial peptides might be found is far
        larger than any laboratory can screen, which makes computational triage
        part of the experimental pipeline rather than an accessory to it. A tool
        that performs that triage is only useful to the extent that other people
        can run it, check it and disagree with it.</p>

        <p>That is why every part of this project is in the open. The training
        code, the datasets, the feature extraction, the model files and the web
        server are in one public repository under a permissive licence. The
        benchmark figures on this page come from a held-out set that ships with
        the repository, so anyone can reproduce them or show that they do not
        hold. A prediction you cannot audit is an opinion with a decimal point.</p>

        <p>The FAIR principles are the practical form of that commitment.
        <strong>Findable</strong>: the software is deposited on PyPI and GitHub,
        and carries a machine-readable <code>CITATION.cff</code> so that a
        citation can be resolved without reading the page. <strong>Accessible</strong>: the web tool needs
        no account, no institutional login and no payment, and the package
        installs with one command. <strong>Interoperable</strong>: input is FASTA
        and output is CSV, two formats every other tool in the pipeline already
        reads. <strong>Reusable</strong>: the licence permits reuse and
        modification, the descriptors are documented, and the training procedure
        is a script rather than a paragraph in a methods section.</p>

        <p>Two commitments follow from this and constrain how the tool is built.
        Sequences submitted here are processed in memory and are never stored:
        unpublished sequence data is the most valuable thing a research group
        holds, and asking someone to upload it to a server that keeps it is
        asking too much. And the tool stays free to run, because a triage step
        behind a paywall stops being triage for exactly the laboratories that
        most need it.</p>

        <p>Open development is not a licence file. It is answering issues,
        publishing the failure cases along with the benchmark, and treating a
        report that the model is wrong as the most useful message the project
        receives.</p>
      </div>
    </div>


    <div class="metrics-band step-2">
      <div class="metrics-label">Tool decay</div>
      <div class="card-glass prose-block prose-justify">
        <p>Bioinformatics web tools stop working. Kern, Fehlmann and Keller
        (2020, <a href="https://doi.org/10.1093/nar/gkaa1125" target="_blank" rel="noopener">doi:10.1093/nar/gkaa1125</a>)
        monitored 2,396 tools published from 2010 onward over 133 days and found
        25.7% unreachable at first access. Availability tracks age almost
        linearly: tools published in 2019 and 2020 were around 90% available,
        those from 2010 around 50%. When the authors of 47 broken recent tools
        were contacted, 51.1% were restored, which means half of those failures
        were not technical problems but abandoned maintenance.</p>

        <p>The same pattern appeared inside this project. Building the external
        benchmark, 20 published antimicrobial peptide predictors were identified
        as candidates. Nine of them, 45%, could not be evaluated at all: web
        servers unreachable, DNS resolution failing, or no public code release to
        run locally. Their papers span 2012 to 2023, and the check was made in
        March 2026. The remaining 11 configurations were benchmarked against the
        same independent set of 4,736 sequences.</p>
      </div>
    </div>

    <div class="metrics-band step-2">
      <div class="metrics-label">This project's external benchmark, March 2026</div>
      <div class="metrics-grid metrics-4">
        <div class="card-glass metric"><span class="num metric-val">20</span><span class="metric-lbl">Candidate tools</span></div>
        <div class="card-glass metric"><span class="num metric-val">9</span><span class="metric-lbl">Could not be run</span></div>
        <div class="card-glass metric"><span class="num metric-val">45%</span><span class="metric-lbl">Of the candidates</span></div>
        <div class="card-glass metric"><span class="num metric-val">2012&ndash;2023</span><span class="metric-lbl">In publications</span></div>
      </div>
    </div>

    <div class="metrics-band step-2">
      <div class="metrics-label">Published survey of 2,396 bioinformatics web services</div>
      <div class="metrics-grid metrics-3">
        <div class="card-glass metric"><span class="num metric-val">25.7%</span><span class="metric-lbl">Unreachable at first access</span></div>
        <div class="card-glass metric"><span class="num metric-val">50%</span><span class="metric-lbl">Availability of 2010 tools</span></div>
        <div class="card-glass metric"><span class="num metric-val">51.1%</span><span class="metric-lbl">Restored after contact</span></div>
      </div>
    </div>

    <div class="metrics-band step-2">
      <div class="metrics-label">What follows from it</div>
      <div class="card-glass prose-block prose-justify">
        <p>A tool that exists only as a web server dies with the server. This one
        is distributed three ways on purpose: the web page, a command line
        program and a Python package on PyPI, with the training code, the
        datasets and the model files in the repository. If this page goes down,
        <code>pip install ampidentifier</code> still reproduces every number
        printed here, and the repository can be forked by anyone who wants to
        keep it alive.</p>
        <p>Reference: Kern, F., Fehlmann, T. &amp; Keller, A. (2020). On the
        lifetime of bioinformatics web services. <em>Nucleic Acids Research</em>
        48(22), 12523&ndash;12533.
        <a href="https://doi.org/10.1093/nar/gkaa1125" target="_blank" rel="noopener">doi:10.1093/nar/gkaa1125</a></p>
      </div>
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
        <p>On the independent benchmark set of 4,736 sequences the ensemble
        reaches AUC-ROC 0.950, MCC 0.742, sensitivity 94.9% and specificity
        78.4%. Single-model modes are available and score lower on every one of
        those four.</p>
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
      <div class="metrics-label">Origin</div>
      <div class="card-glass prose-block prose-justify">
        <p>AMPidentifier began as a minimum viable product written by Jo&atilde;o
        Pacifico Bezerra Neto in the 2020s. That prototype is where the tool
        comes from: it set the problem, showed that sequence-derived descriptors
        carried enough signal to classify antimicrobial peptides, and made the
        case for building the rest.</p>
        <p>What is distributed today was rebuilt from that starting point. The
        descriptor set, the training pipeline, the ensemble, the external
        benchmark and the three distribution channels are a later and separate
        implementation, but the idea is his.</p>
      </div>
    </div>

    <div class="metrics-band step-2">
      <div class="metrics-label">Maintainer</div>
      <div class="card-glass lead-card">
        %(photo)s
        <div class="lead-text prose-justify">
          <div class="lead-name">Madson Allan de Luna Arag&atilde;o</div>
          <div class="lead-role">Lead developer and maintainer</div>
          <p>Wrote the three ways the tool ships, the command line program,
          the Python package on PyPI and this web server, so that none of them
          is a single point of failure for the others.</p>
          <p>Maintenance is the work, not what comes after it. Half the broken
          tools in the survey above came back the moment someone answered an
          email, which is most of the distance between a published tool and a
          working one. Keeps the repository current, answers the issues, keeps
          the server up, and holds the constraints the tool is built against: no
          sequence is stored, no account is required, and nothing is charged.</p>
          <div class="lead-education">
            <div class="edu-label">Education</div>
            <ul class="edu-list">
              <li><span class="edu-what">PhD student in Bioinformatics</span><span class="edu-where">Institute of Biological Sciences, UFMG</span><span class="num edu-when">2024&ndash;Now</span></li>
              <li><span class="edu-what">MBA in Software Engineering</span><span class="edu-where">Computer Science Department, USP</span><span class="num edu-when">2025&ndash;Now</span></li>
              <li><span class="edu-what">Specialization in Data Science and Analytics</span><span class="edu-where">Computer Science Department, PUC-Rio</span><span class="num edu-when">2024&ndash;2026</span></li>
              <li><span class="edu-what">MSc in Genetics and Molecular Biology</span><span class="edu-where">Department of Genetics, UFPE</span><span class="num edu-when">2022&ndash;2024</span></li>
              <li><span class="edu-what">BSc in Biomedical Sciences</span><span class="edu-where">Center of Biosciences, UFPE</span><span class="num edu-when">2014&ndash;2021</span></li>
            </ul>
          </div>
          <p class="lead-links">
            <a href="https://orcid.org/0000-0001-5313-3913" target="_blank" rel="noopener">ORCID 0000-0001-5313-3913</a>
            &nbsp;&middot;&nbsp; <a href="https://github.com/madsondeluna" target="_blank" rel="noopener">GitHub</a>
            &nbsp;&middot;&nbsp; <a href="https://madsondeluna.com" target="_blank" rel="noopener">madsondeluna.com</a>
            &nbsp;&middot;&nbsp; <a href="mailto:madsondeluna@gmail.com">madsondeluna@gmail.com</a>
          </p>
        </div>
      </div>
    </div>

    <div class="metrics-band step-2">
      <div class="metrics-label">Authors</div>
      <div class="author-grid">

        <div class="card-glass author">
          <div class="author-name">Madson Allan de Luna Arag&atilde;o</div>
          <div class="author-role">Lead developer and maintainer</div>
          <div class="author-affil">Institute of Biological Sciences, Universidade Federal de Minas Gerais (UFMG), Belo Horizonte<br>Department of Genetics, Universidade Federal de Pernambuco (UFPE), Recife</div>
          <div class="author-links"><a href="https://orcid.org/0000-0001-5313-3913" target="_blank" rel="noopener">ORCID</a></div>
        </div>

        <div class="card-glass author">
          <div class="author-name">Rafael Lucas da Silva</div>
          <div class="author-affil">Department of Genetics, Universidade Federal de Pernambuco (UFPE), Recife</div>
        </div>

        <div class="card-glass author">
          <div class="author-name">Jo&atilde;o Pacifico Bezerra Neto</div>
          <div class="author-role">Original prototype</div>
          <div class="author-affil">Universidade de Pernambuco (UPE), Petrolina</div>
          <div class="author-links"><a href="https://orcid.org/0000-0003-3861-4879" target="_blank" rel="noopener">ORCID</a></div>
        </div>

        <div class="card-glass author">
          <div class="author-name">Carlos Andr&eacute; dos Santos-Silva</div>
          <div class="author-affil">Centro Universit&aacute;rio CESMAC, Macei&oacute;</div>
        </div>

        <div class="card-glass author">
          <div class="author-name">Denys Ewerton da Silva Santos</div>
          <div class="author-affil">Department of Fundamental Chemistry, Universidade Federal de Pernambuco (UFPE), Recife</div>
        </div>

        <div class="card-glass author">
          <div class="author-name">Ana Maria Benko-Iseppon</div>
          <div class="author-affil">Department of Genetics, Universidade Federal de Pernambuco (UFPE), Recife</div>
          <div class="author-links"><a href="https://orcid.org/0000-0002-0575-3197" target="_blank" rel="noopener">ORCID</a></div>
        </div>

      </div>
    </div>

    <div class="metrics-band step-2">
      <div class="metrics-label">Access</div>
      <div class="card-glass prose-block prose-justify">
        <p class="install"><span class="install-lead">Python package:</span> <code>pip install ampidentifier</code></p>
        <p class="install"><span class="install-lead">Source and command line:</span> <a href="https://github.com/madsondeluna/AMPidentifier" target="_blank" rel="noopener">github.com/madsondeluna/AMPidentifier</a></p>
        <p class="install"><span class="install-lead">Web:</span> <a href="/">the predictor</a> and <a href="/beta">what is coming next</a></p>
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
        <p>de Luna-Arag&atilde;o, M. A., da Silva, R. L., Pacifico Bezerra Neto, J.,
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

  /* o retrato e a coluna de texto dividem uma linha; abaixo do ponto de
     quebra empilham, e a foto encolhe em vez de tomar a largura toda */
  .lead-card {
    display: flex;
    align-items: flex-start;
    gap: var(--space-24);
    padding: var(--space-24);
  }

  .lead-photo {
    flex: 0 0 auto;
    width: var(--photo-sm);
    height: var(--photo-sm);
    border-radius: var(--radius-circle);
    object-fit: cover;
    display: block;
  }

  .lead-text { min-width: 0; }
  .lead-name { font-size: var(--text-16); font-weight: var(--weight-medium); color: var(--text); }
  .lead-role { font-family: var(--font-mono); font-size: var(--text-12); color: var(--muted); margin-bottom: var(--space-10); }
  .lead-text > p + p, .lead-role + p { margin-top: var(--space-10); }
  .lead-links { font-size: var(--text-13); }

  /* formacao: tres colunas, com o ano em coluna propria e alinhado a
     direita, porque ano se compara com ano e nao com o fim do texto que
     vem antes dele. A lista nao herda a justificacao da prosa: linha
     curta justificada abre buraco entre as palavras. */
  .lead-education { margin-top: var(--space-16); text-align: left; }
  .edu-label { font-family: var(--font-mono); font-size: var(--text-11); color: var(--muted); margin-bottom: var(--space-6); }
  .edu-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--space-4); }
  .edu-list li { display: grid; grid-template-columns: 1fr auto auto; gap: var(--space-10); align-items: baseline; font-size: var(--text-13); }
  .edu-what { color: var(--text); }
  .edu-where { color: var(--muted); font-size: var(--text-12); }
  .edu-when { color: var(--muted); font-size: var(--text-12); white-space: nowrap; }

  @media (max-width: 768px) {
    .edu-list li { grid-template-columns: 1fr; gap: 0; }
  }

  /* seis autores em duas colunas: uma coluna so faz seis cartoes de duas
     linhas cada e a lista fica mais alta que o manifesto que a precede */
  .metrics-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--space-8); }

  .author-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: var(--space-12);
  }

  .author { padding: var(--space-16); display: flex; flex-direction: column; gap: var(--space-6); }
  .author-name { font-size: var(--text-15); color: var(--text); }
  .author-role { font-family: var(--font-mono); font-size: var(--text-11); color: var(--muted); }
  .author-affil { font-size: var(--text-12); color: var(--muted); line-height: var(--leading-snug); }
  .author-links { font-family: var(--font-mono); font-size: var(--text-11); }

  @media (max-width: 768px) {
    .metrics-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--space-8); }

  .author-grid { grid-template-columns: 1fr; }
    .lead-card { flex-direction: column; align-items: flex-start; }
  }
"""

PAGE = page(
    title='About | AMPidentifier',
    description=('How AMPidentifier predicts antimicrobial peptides: the voting '
                 'ensemble, the 22 descriptors, the benchmark figures, the scope '
                 'of the training data and the privacy terms.'),
    path='/about',
    body=BODY.replace('%(photo)s', PHOTO_TAG),
    css=CSS,
)
