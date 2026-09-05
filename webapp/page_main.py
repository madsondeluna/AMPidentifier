"""Main front end: the Pure Design page, served at /.

This is the page the site serves at the root. The previous markup is
still in app.py as PAGE, unrouted, so the old layout can be restored by
pointing / back at it.

O head, a folha, a barra de cima, a barra de baixo e o modal moram em
page_shell.py e sao os mesmos de /about e /suggestions. Aqui fica so o
miolo desta pagina e o javascript dela.
"""

from webapp.page_shell import page

BODY = """  <header>
    <div>
      <!-- a marca e a imagem; o titulo da pagina existe para leitor de tela
           e para a hierarquia do documento, que nao tinha nenhum h1 -->
      <h1 class="sr-only">AMPidentifier, antimicrobial peptide prediction</h1>
    </div>

    <div class="metrics-band step-1">
     <div class="card-glass intro">
      <p class="sub prose-justify"><strong>AMPidentifier</strong> is a toolkit for antimicrobial peptide prediction using ensemble machine learning.</p>

      <div class="install-stack">
        <p class="install"><span class="install-lead">For <a href="https://pypi.org/project/ampidentifier/" target="_blank">PyPI</a>:</span> <code>pip install ampidentifier</code></p>
        <p class="install"><span class="install-lead">For terminal use:</span> <a href="https://github.com/madsondeluna/AMPIdentifier" target="_blank">CLI version</a></p>
        <p class="install"><span class="install-lead">In testing:</span> <a href="/beta">what is coming next</a></p>
      <p class="install"><span class="install-lead">Previous layout:</span> <a href="/legacy">legacy version</a></p>
      </div>
     </div>
    </div>
  </header>

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

    <div id="results" class="motion-lines"></div>

    <div class="surface share-section step-2">
      <div class="share-inner">
        <div class="share-heading">Find AMPidentifier useful?</div>
        <div class="share-actions">
          <button class="pill" onclick="copyLink()" id="copyLinkBtn">Copy link</button>
          <button class="pill" onclick="toggleShareForm()" id="shareEmailBtn">Share by email</button>
        </div>
      </div>
      <div class="share-url-box mono motion-dropdown" id="shareUrlBox"></div>
      <div class="share-form motion-dropdown" id="shareForm">
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
  </main>

  <footer class="step-2">
    <div>
      <div class="usage-map-title">Where AMPidentifier is being used</div>
      <div id="usageMap"></div>
    </div>

  <div class="metrics-band step-2">
    <div class="metrics-label">In testing</div>
    <div class="card-glass changelog">
      <p>This round changes the interface only. Models, thresholds and predictions are the same as the stable version.</p>
      <p class="changelog-body prose-justify">The front end was rebuilt on a token-based design system: one type scale, one spacing scale and a single set of colour tokens shared by every component, with the layout on a single column and a concentric radius ladder. Controls and panels became glass surfaces with backdrop-filter, keyboard focus rings and reduced-motion fallbacks, the usage map became inline SVG instead of a tile layer, and the result panel carries its state in the URL.</p>
      <p class="changelog-body prose-justify">Coming soon: a new batch of trained models will reach the beta before the stable version, and a prediction mode built on a protein language model (PLLM) goes into testing here.</p>
    </div>
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
      <p>Developer: <a href="mailto:madsondeluna@gmail.com">madsondeluna@gmail.com</a> &nbsp;·&nbsp; <a href="https://madsondeluna.com" target="_blank">madsondeluna.com</a> &nbsp;·&nbsp; <button class="feedback-link hit" onclick="openFeedback()">Report issue or suggestion</button> &nbsp;·&nbsp; <span class="version">v{{ version }}</span></p>
    </div>
  </footer>"""

JS = """const EXAMPLE = [
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
    // ones stay visible when the SVG shrinks on narrow viewports.
    // the hit circle has its own floor, read from the interaction token:
    // the drawn ring carries the datum, the invisible one carries the hand.
    const hitFloorPx = function() {
      const root = getComputedStyle(document.documentElement);
      const key = window.matchMedia('(pointer: coarse)').matches ? '--hit-min-touch' : '--hit-min';
      return (parseFloat(root.getPropertyValue(key)) || 0) / 2;
    };
    const sizeRings = function() {
      const unit = (svg.getBoundingClientRect().width || world.w) / world.w;
      if (!unit) return;
      const floor = 4.5 / unit;
      const hitFloor = hitFloorPx() / unit;
      rings.forEach(function(item) {
        const r = Math.max(item.r, floor);
        item.node.setAttribute('r', r);
        item.gloss.setAttribute('r', r);
        item.hit.setAttribute('r', Math.max(r, hitFloor));
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
      const hit  = el('circle', { cx: x, cy: y, r: r, class: 'ring-hit' }, g);
      rings.push({ node: ring, gloss: lit, hit: hit, r: r });
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
/* O escalonamento so corre no resultado final. O esqueleto entra sem
   replay: dois escalonamentos seguidos no mesmo painel leem como falha
   de carregamento, nao como sequencia. */
function revealResults(replay) {
  const box = document.getElementById('results');
  if (replay) { box.classList.remove('is-open'); void box.offsetWidth; }
  box.classList.add('is-open');
}

function showEmptyResults() {
  document.getElementById('results').innerHTML =
    '<div class="empty">' +
      '<div><div class="empty-head">No predictions yet</div>' +
      '<div>Paste FASTA sequences above, or start from the example.</div></div>' +
    '</div>';
  revealResults(true);
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
  revealResults(false);
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

  revealResults(true);
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
    motionOpen(box);
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy link'; motionClose(box); }, 2500);
  }).catch(() => {
    box.textContent = url;
    motionOpen(box);
  });
}

function toggleShareForm() {
  const form = document.getElementById('shareForm');
  if (form.classList.contains('open')) { motionClose(form); return; }
  motionOpen(form);
  document.getElementById('shareFriendEmail').focus();
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
}"""

PAGE = page(
    title='AMPidentifier | Antimicrobial Peptide Prediction Tool',
    description=('AMPidentifier is a free web tool for antimicrobial peptide (AMP) '
                 'prediction using machine learning ensemble models. Submit FASTA '
                 'sequences and classify AMPs in seconds.'),
    path='/',
    body=BODY,
    js=JS,
)
