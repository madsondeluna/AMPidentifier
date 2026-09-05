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
          <ul class="lead-links">
            <li>
              <svg class="icon icon-sm" aria-hidden="true"><use href="/pure/icons.svg#tag"></use></svg>
              <a href="https://orcid.org/0000-0001-5313-3913" target="_blank" rel="noopener">0000-0001-5313-3913</a>
            </li>
            <li>
              <svg class="icon icon-sm" aria-hidden="true"><use href="/pure/icons.svg#branch"></use></svg>
              <a href="https://github.com/madsondeluna" target="_blank" rel="noopener">github.com/madsondeluna</a>
            </li>
            <li>
              <svg class="icon icon-sm" aria-hidden="true"><use href="/pure/icons.svg#link"></use></svg>
              <a href="https://madsondeluna.com" target="_blank" rel="noopener">madsondeluna.com</a>
            </li>
            <li>
              <svg class="icon icon-sm" aria-hidden="true"><use href="/pure/icons.svg#mail"></use></svg>
              <a href="mailto:madsondeluna@gmail.com">madsondeluna@gmail.com</a>
            </li>
          </ul>
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
  /* Duas colunas, icone numa coluna propria de largura fixa: assim os
     quatro enderecos comecam na mesma vertical e o icone nao empurra o
     texto conforme muda de desenho. O icone e --muted e o endereco e
     tinta cheia, porque quem se le e o endereco; o icone so diz de que
     tipo ele e, e por isso leva aria-hidden. */
  .lead-links {
    list-style: none;
    margin: var(--space-16) 0 0;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-8) var(--space-24);
    font-size: var(--text-13);
  }

  .lead-links li {
    display: grid;
    grid-template-columns: var(--space-20) 1fr;
    align-items: center;
    gap: var(--space-8);
    min-width: 0;
  }

  .lead-links .icon { color: var(--muted); }
  .lead-links a { color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  @media (max-width: 768px) {
    .lead-links { grid-template-columns: 1fr; }
  }

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

# ---------------------------------------------------------------------------
# Versao em portugues.
#
# Nao se traduz: nome de instituicao, nome de autor, titulo de artigo, nome
# de periodico, DOI, sigla de metrica, numero de registro no INPI e a
# citacao. Sao registros, e traduzir registro quebra o casamento com a
# fonte que o declara.
#
# A prosa longa e trocada por PARAGRAFO inteiro, ancorada numa frase que so
# aparece nele: casar espaco em branco em texto de dez linhas quebra ao
# primeiro reflow do arquivo.
# ---------------------------------------------------------------------------

import re as _re

PARAGRAPHS = [
    ('AMPidentifier classifies peptide sequences',
     'O AMPidentifier classifica sequ&ecirc;ncias de pept&iacute;deos como antimicrobianas ou n&atilde;o '
     'antimicrobianas a partir da sequ&ecirc;ncia prim&aacute;ria de amino&aacute;cidos apenas. Ele roda como '
     'ferramenta web, programa de linha de comando e pacote Python, sobre os mesmos modelos e os '
     'mesmos limiares nos tr&ecirc;s.'),

    ('Antimicrobial resistance kills people now',
     'A resist&ecirc;ncia antimicrobiana mata gente hoje e vai matar mais. O espa&ccedil;o de sequ&ecirc;ncias '
     'onde novos pept&iacute;deos antimicrobianos podem estar &eacute; muito maior do que qualquer '
     'laborat&oacute;rio consegue rastrear, o que faz da triagem computacional parte do pipeline '
     'experimental e n&atilde;o um acess&oacute;rio dele. Uma ferramenta que faz essa triagem s&oacute; serve na '
     'medida em que outras pessoas conseguem execut&aacute;-la, conferi-la e discordar dela.'),

    ('That is why every part of this project is in the open',
     '&Eacute; por isso que cada parte deste projeto est&aacute; aberta. O c&oacute;digo de treino, os conjuntos de '
     'dados, a extra&ccedil;&atilde;o de descritores, os arquivos de modelo e o servidor web est&atilde;o num '
     'reposit&oacute;rio p&uacute;blico sob licen&ccedil;a permissiva. Os n&uacute;meros do benchmark nesta p&aacute;gina saem de '
     'um conjunto de teste separado que acompanha o reposit&oacute;rio, ent&atilde;o qualquer um pode '
     'reproduzi-los ou mostrar que eles n&atilde;o se sustentam. Uma predi&ccedil;&atilde;o que voc&ecirc; n&atilde;o pode '
     'auditar &eacute; uma opini&atilde;o com casa decimal.'),

    ('The FAIR principles are the practical form',
     'Os princ&iacute;pios FAIR s&atilde;o a forma pr&aacute;tica desse compromisso. '
     '<strong>Localiz&aacute;vel</strong>: o software est&aacute; depositado no PyPI e no GitHub, e traz um '
     '<code>CITATION.cff</code> leg&iacute;vel por m&aacute;quina, de modo que a cita&ccedil;&atilde;o se resolve sem ler a '
     'p&aacute;gina. <strong>Acess&iacute;vel</strong>: a ferramenta web n&atilde;o pede conta, nem login '
     'institucional, nem pagamento, e o pacote instala com um comando. '
     '<strong>Interoper&aacute;vel</strong>: a entrada &eacute; FASTA e a sa&iacute;da &eacute; CSV, dois formatos que toda '
     'outra ferramenta do pipeline j&aacute; l&ecirc;. <strong>Reutiliz&aacute;vel</strong>: a licen&ccedil;a permite reuso '
     'e modifica&ccedil;&atilde;o, os descritores est&atilde;o documentados, e o procedimento de treino &eacute; um '
     'script e n&atilde;o um par&aacute;grafo numa se&ccedil;&atilde;o de m&eacute;todos.'),

    ('Two commitments follow from this',
     'Dois compromissos decorrem disso e limitam como a ferramenta &eacute; constru&iacute;da. Sequ&ecirc;ncias '
     'enviadas aqui s&atilde;o processadas em mem&oacute;ria e nunca armazenadas: dado de sequ&ecirc;ncia n&atilde;o '
     'publicado &eacute; a coisa mais valiosa que um grupo de pesquisa guarda, e pedir que algu&eacute;m o '
     'envie para um servidor que o retenha &eacute; pedir demais. E a ferramenta continua gratuita, '
     'porque uma etapa de triagem atr&aacute;s de paywall deixa de ser triagem justamente para os '
     'laborat&oacute;rios que mais precisam dela.'),

    ('Open development is not a licence file',
     'Desenvolvimento aberto n&atilde;o &eacute; um arquivo de licen&ccedil;a. &Eacute; responder issues, publicar os '
     'casos de falha junto com o benchmark, e tratar um relato de que o modelo errou como a '
     'mensagem mais &uacute;til que o projeto recebe.'),

    ('Bioinformatics web tools stop working',
     'Ferramentas web de bioinform&aacute;tica param de funcionar. Kern, Fehlmann e Keller '
     '(2020, <a href="https://doi.org/10.1093/nar/gkaa1125" target="_blank" rel="noopener">doi:10.1093/nar/gkaa1125</a>) '
     'monitoraram 2.396 ferramentas publicadas a partir de 2010 ao longo de 133 dias e '
     'encontraram 25,7% inalcan&ccedil;&aacute;veis no primeiro acesso. A disponibilidade acompanha a idade '
     'quase linearmente: as publicadas em 2019 e 2020 estavam por volta de 90% dispon&iacute;veis, as '
     'de 2010 por volta de 50%. Quando os autores de 47 ferramentas recentes quebradas foram '
     'contatados, 51,1% voltaram ao ar, o que significa que metade daquelas falhas n&atilde;o era '
     'problema t&eacute;cnico, era manuten&ccedil;&atilde;o abandonada.'),

    ('The same pattern appeared inside this project',
     'O mesmo padr&atilde;o apareceu dentro deste projeto. Ao montar o benchmark externo, 20 '
     'preditores de pept&iacute;deos antimicrobianos publicados foram identificados como candidatos. '
     'Nove deles, 45%, n&atilde;o puderam ser avaliados: servidores inalcan&ccedil;&aacute;veis, falha de resolu&ccedil;&atilde;o '
     'de DNS, ou nenhuma libera&ccedil;&atilde;o p&uacute;blica de c&oacute;digo para rodar localmente. Os artigos deles v&atilde;o '
     'de 2012 a 2023, e a verifica&ccedil;&atilde;o foi feita em mar&ccedil;o de 2026. As 11 configura&ccedil;&otilde;es '
     'restantes foram avaliadas contra o mesmo conjunto independente de 4.736 sequ&ecirc;ncias.'),

    ('A tool that exists only as a web server dies',
     'Uma ferramenta que existe s&oacute; como servidor web morre com o servidor. Esta &eacute; distribu&iacute;da '
     'de tr&ecirc;s formas de prop&oacute;sito: a p&aacute;gina web, um programa de linha de comando e um pacote '
     'Python no PyPI, com o c&oacute;digo de treino, os conjuntos de dados e os arquivos de modelo no '
     'reposit&oacute;rio. Se esta p&aacute;gina cair, <code>pip install ampidentifier</code> ainda reproduz '
     'cada n&uacute;mero impresso aqui, e o reposit&oacute;rio pode ser bifurcado por quem quiser '
     'mant&ecirc;-lo vivo.'),

    ('The default mode is a soft-voting ensemble',
     'O modo padr&atilde;o &eacute; um ensemble por vota&ccedil;&atilde;o suave de cinco classificadores: random forest, '
     'support vector machine, gradient boosting, XGBoost e LightGBM. Cada sequ&ecirc;ncia &eacute; reduzida '
     'a 22 descritores f&iacute;sico-qu&iacute;micos e composicionais, entre eles carga l&iacute;quida, '
     'hidrofobicidade, momento hidrof&oacute;bico, ponto isoel&eacute;trico, &iacute;ndice alif&aacute;tico, &iacute;ndice de '
     'instabilidade e composi&ccedil;&atilde;o de amino&aacute;cidos.'),

    ('On the independent benchmark set of 4,736',
     'No conjunto de benchmark independente de 4.736 sequ&ecirc;ncias o ensemble alcan&ccedil;a AUC-ROC '
     '0,950, MCC 0,742, sensibilidade 94,9% e especificidade 78,4%. Modos de modelo &uacute;nico est&atilde;o '
     'dispon&iacute;veis e pontuam menos em cada uma dessas quatro.'),

    ('Training data comes from experimentally validated',
     'Os dados de treino v&ecirc;m de pept&iacute;deos antimicrobianos validados experimentalmente em bases '
     'p&uacute;blicas, contra sequ&ecirc;ncias n&atilde;o antimicrobianas. A cobertura inclui pept&iacute;deos '
     'antibacterianos, antif&uacute;ngicos, antivirais e outros pept&iacute;deos de defesa do hospedeiro. Uma '
     'prote&iacute;na cuja fun&ccedil;&atilde;o prim&aacute;ria n&atilde;o &eacute; antimicrobiana ainda pode carregar caracter&iacute;sticas '
     'antimicrobianas em regi&otilde;es espec&iacute;ficas da sua sequ&ecirc;ncia, ent&atilde;o uma chamada positiva &eacute; uma '
     'hip&oacute;tese para testar, n&atilde;o uma medida.'),

    ('AMPidentifier began as a minimum viable product',
     'O AMPidentifier come&ccedil;ou como um produto m&iacute;nimo vi&aacute;vel escrito por Jo&atilde;o Pacifico Bezerra '
     'Neto nos anos 2020. Aquele prot&oacute;tipo &eacute; de onde a ferramenta vem: ele fixou o problema, '
     'mostrou que descritores derivados da sequ&ecirc;ncia carregavam sinal suficiente para classificar '
     'pept&iacute;deos antimicrobianos, e justificou construir o resto.'),

    ('What is distributed today was rebuilt',
     'O que se distribui hoje foi reconstru&iacute;do a partir daquele ponto de partida. O conjunto de '
     'descritores, o pipeline de treino, o ensemble, o benchmark externo e os tr&ecirc;s canais de '
     'distribui&ccedil;&atilde;o s&atilde;o uma implementa&ccedil;&atilde;o posterior e separada, mas a ideia &eacute; dele.'),

    ('Wrote the three ways the tool ships',
     'Escreveu as tr&ecirc;s formas pelas quais a ferramenta &eacute; distribu&iacute;da, o programa de linha de '
     'comando, o pacote Python no PyPI e este servidor web, de modo que nenhuma delas seja ponto '
     '&uacute;nico de falha para as outras.'),

    ('Maintenance is the work, not what comes after it',
     'Manuten&ccedil;&atilde;o &eacute; o trabalho, n&atilde;o o que vem depois dele. Metade das ferramentas quebradas do '
     'levantamento acima voltou no momento em que algu&eacute;m respondeu um email, o que &eacute; a maior '
     'parte da dist&acirc;ncia entre uma ferramenta publicada e uma ferramenta que funciona. Mant&eacute;m o '
     'reposit&oacute;rio em dia, responde as issues, mant&eacute;m o servidor no ar, e sustenta as restri&ccedil;&otilde;es '
     'contra as quais a ferramenta &eacute; constru&iacute;da: nenhuma sequ&ecirc;ncia &eacute; armazenada, nenhuma conta '
     '&eacute; exigida, e nada &eacute; cobrado.'),

    ('Sequences are processed in memory',
     'Sequ&ecirc;ncias s&atilde;o processadas em mem&oacute;ria e n&atilde;o s&atilde;o armazenadas. A transfer&ecirc;ncia &eacute; '
     'criptografada com HTTPS/TLS. Nenhuma conta &eacute; exigida e nenhum dado de sequ&ecirc;ncia &eacute; '
     'compartilhado com terceiros.'),
]

LABELS = [
    ('>About AMPidentifier</h1>', '>Sobre o AMPidentifier</h1>'),
    ('>Manifesto</div>', '>Manifesto</div>'),
    ('>Tool decay</div>', '>Ferramentas que somem</div>'),
    (">This project's external benchmark, March 2026</div>",
     '>Benchmark externo deste projeto, mar&ccedil;o de 2026</div>'),
    ('>Candidate tools</span>', '>Ferramentas candidatas</span>'),
    ('>Could not be run</span>', '>N&atilde;o puderam rodar</span>'),
    ('>Of the candidates</span>', '>Das candidatas</span>'),
    ('>In publications</span>', '>Em publica&ccedil;&otilde;es</span>'),
    ('>Published survey of 2,396 bioinformatics web services</div>',
     '>Levantamento publicado de 2.396 servi&ccedil;os web de bioinform&aacute;tica</div>'),
    ('>Unreachable at first access</span>', '>Inalcan&ccedil;&aacute;veis no primeiro acesso</span>'),
    ('>Availability of 2010 tools</span>', '>Disponibilidade das de 2010</span>'),
    ('>Restored after contact</span>', '>Restauradas ap&oacute;s contato</span>'),
    ('>What follows from it</div>', '>O que decorre disso</div>'),
    ('>Model</div>', '>Modelo</div>'),
    ('>Benchmark, voting ensemble</div>', '>Benchmark, ensemble por vota&ccedil;&atilde;o</div>'),
    ('>Sensitivity</span>', '>Sensibilidade</span>'),
    ('>Specificity</span>', '>Especificidade</span>'),
    ('>Scope</div>', '>Abrang&ecirc;ncia</div>'),
    ('>Origin</div>', '>Origem</div>'),
    ('>Maintainer</div>', '>Mantenedor</div>'),
    ('>Lead developer and maintainer</div>', '>Desenvolvedor principal e mantenedor</div>'),
    ('>Original prototype</div>', '>Prot&oacute;tipo original</div>'),
    ('>Education</div>', '>Forma&ccedil;&atilde;o</div>'),
    ('>PhD student in Bioinformatics</span>', '>Doutorando em Bioinform&aacute;tica</span>'),
    ('>MBA in Software Engineering</span>', '>MBA em Engenharia de Software</span>'),
    ('>Specialization in Data Science and Analytics</span>',
     '>Especializa&ccedil;&atilde;o em Data Science e Analytics</span>'),
    ('>MSc in Genetics and Molecular Biology</span>',
     '>Mestrado em Gen&eacute;tica e Biologia Molecular</span>'),
    ('>BSc in Biomedical Sciences</span>', '>Bacharelado em Ci&ecirc;ncias Biom&eacute;dicas</span>'),
    ('>Authors</div>', '>Autores</div>'),
    ('>Access</div>', '>Acesso</div>'),
    ('>Python package:</span>', '>Pacote Python:</span>'),
    ('>Source and command line:</span>', '>C&oacute;digo e linha de comando:</span>'),
    ('>Web:</span>', '>Web:</span>'),
    ('>the predictor</a>', '>o preditor</a>'),
    ('>what is coming next</a>', '>o que vem por a&iacute;</a>'),
    ('>Privacy</div>', '>Privacidade</div>'),
    ('>Citation</div>', '>Cita&ccedil;&atilde;o</div>'),
    ('Registered with the', 'Registrada no'),
    ('under No.', 'sob o n&uacute;mero'),
    ('property of the', 'propriedade da'),
]


def _swap(body, anchor, novo):
    """Troca o paragrafo inteiro que contem a ancora."""
    m = _re.search(r'<p([^>]*)>((?:(?!</p>).)*?' + _re.escape(anchor) + r'(?:(?!</p>).)*?)</p>',
                   body, _re.S)
    assert m, anchor
    return body[:m.start()] + '<p%s>%s</p>' % (m.group(1), novo) + body[m.end():]


BODY_PT = BODY
for _anchor, _novo in PARAGRAPHS:
    BODY_PT = _swap(BODY_PT, _anchor, _novo)
for _a, _b in LABELS:
    assert _a in BODY_PT, _a
    BODY_PT = BODY_PT.replace(_a, _b, 1)
BODY_PT = BODY_PT.replace('href="/beta"', 'href="/pt/beta"').replace('href="/">o preditor', 'href="/pt">o preditor')


PAGE = page(
    title='About | AMPidentifier',
    description=('How AMPidentifier predicts antimicrobial peptides: the voting '
                 'ensemble, the 22 descriptors, the benchmark figures, the scope '
                 'of the training data and the privacy terms.'),
    path='/about',
    body=BODY.replace('%(photo)s', PHOTO_TAG),
    css=CSS,
)

PAGE_PT = page(
    title='Sobre | AMPidentifier',
    description=('Como o AMPidentifier prev\u00ea pept\u00eddeos antimicrobianos: o ensemble '
                 'por vota\u00e7\u00e3o, os 22 descritores, os n\u00fameros do benchmark, a '
                 'abrang\u00eancia dos dados de treino e os termos de privacidade.'),
    path='/about',
    body=BODY_PT.replace('%(photo)s', PHOTO_TAG),
    css=CSS,
    lang='pt',
)
