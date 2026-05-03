"""
AMPidentifier Web Portal — v2.0
"""
import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import threading
import urllib.request
import uuid

from flask import Flask, make_response, request, jsonify, render_template_string
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amp_identifier.core import run_prediction_pipeline
from amp_identifier.data_io import load_fasta_sequences

VERSION = "2.0.0"

def _send_telegram(text):
    token   = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        return
    def _post():
        try:
            payload = json.dumps({'chat_id': chat_id, 'text': text}).encode()
            req = urllib.request.Request(
                f'https://api.telegram.org/bot{token}/sendMessage',
                data=payload,
                headers={'Content-Type': 'application/json'},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
    threading.Thread(target=_post, daemon=True).start()

_db_lock = threading.Lock()

def _db_path():
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stats.db')
    return os.environ.get('STATS_DB', default)

def _conn():
    c = sqlite3.connect(_db_path(), check_same_thread=False)
    c.execute('PRAGMA journal_mode=WAL')
    return c

def init_db():
    with _db_lock:
        c = _conn()
        c.execute('CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value INTEGER DEFAULT 0)')
        c.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY)')
        for k in ('total_sequences', 'total_runs', 'unique_sessions'):
            c.execute('INSERT OR IGNORE INTO stats VALUES (?, 0)', (k,))
        c.commit()
        c.close()

def increment_stats(seq_count, session_id):
    with _db_lock:
        c = _conn()
        c.execute('UPDATE stats SET value = value + ? WHERE key = ?', (seq_count, 'total_sequences'))
        c.execute('UPDATE stats SET value = value + 1 WHERE key = ?', ('total_runs',))
        if not c.execute('SELECT 1 FROM sessions WHERE id = ?', (session_id,)).fetchone():
            c.execute('INSERT OR IGNORE INTO sessions VALUES (?)', (session_id,))
            c.execute('UPDATE stats SET value = value + 1 WHERE key = ?', ('unique_sessions',))
        c.commit()
        c.close()

def get_stats():
    with _db_lock:
        c = _conn()
        rows = c.execute('SELECT key, value FROM stats').fetchall()
        c.close()
        return {k: v for k, v in rows}

app = Flask(__name__, static_folder='img', static_url_path='/img')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
init_db()

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AMPidentifier</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:ital,wght@0,300;0,400;0,500;0,700;1,400&display=swap" rel="stylesheet">
<style>
  html { font-size: 17px; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Roboto Mono', monospace; background: #ffffff; color: #1a1a1a; min-height: 100vh; padding: 28px 24px; }
  .wrap { max-width: 760px; margin: 0 auto; }
  .title-row { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
  h1 { font-size: 1.4rem; font-weight: normal; letter-spacing: 0.1em; color: #0f0f0f; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #ddd; flex-shrink: 0; transition: background 0.4s; cursor: default; }
  .status-dot.online  { background: #059669; }
  .status-dot.offline { background: #dc2626; }
  .sub { font-size: 0.78rem; color: #888; margin-bottom: 10px; }
  .stats-section { margin-top: 12px; margin-bottom: 16px; text-align: center; }
  .stats-section-label { font-size: 0.65rem; color: #ccc; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 20px; }
  .stats-grid { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: center; gap: 20px 32px; }
  .stats-item { display: flex; flex-direction: row; align-items: center; gap: 10px; }
  .stats-val { font-size: 1.8rem; font-weight: 600; color: #1a1a1a; font-variant-numeric: tabular-nums; line-height: 1; flex-shrink: 0; }
  .stats-lbl { font-size: 0.62rem; color: #bbb; text-transform: uppercase; letter-spacing: 0.08em; text-align: left; max-width: 120px; line-height: 1.3; }
  .notice { font-size: 0.75rem; color: #999; border-left: 2px solid #ddd; padding: 6px 12px; margin-bottom: 18px; line-height: 1.6; }
  .notice a { color: #555; text-decoration: underline; }
  .notice a:hover { color: #111; }
  footer { margin-top: 32px; padding-top: 24px; border-top: 1px solid #e8e8e8; font-size: 0.63rem; color: #aaa; line-height: 1.8; text-align: justify; }
  footer a { color: #999; text-decoration: underline; }
  footer a:hover { color: #333; }
  .label-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
  label { font-size: 0.75rem; color: #999; letter-spacing: 0.08em; text-transform: uppercase; }
  .seq-counter { font-size: 0.72rem; color: #bbb; }
  textarea {
    width: 100%; height: 120px; background: #f7f7f7; border: 1px solid #e0e0e0;
    color: #1a1a1a; font-family: 'Roboto Mono', monospace; font-size: 0.82rem;
    padding: 14px; resize: vertical; outline: none; border-radius: 4px;
  }
  textarea:focus { border-color: #bbb; }
  .validation-err { font-size: 0.73rem; color: #dc2626; margin-top: 6px; min-height: 16px; }
  .upload-row { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
  .upload-btn {
    background: #555555; color: #ffffff; border: none;
    font-size: 0.82rem; padding: 10px 28px; font-weight: normal;
    font-family: 'Roboto Mono', monospace; cursor: pointer; border-radius: 4px;
  }
  .upload-btn:hover { background: #444444; }
  #fileInput { display: none; }
  .row { display: flex; gap: 12px; margin-top: 12px; align-items: center; flex-wrap: nowrap; }
  select {
    background: #f7f7f7; border: 1px solid #e0e0e0; color: #1a1a1a;
    font-family: 'Roboto Mono', monospace; font-size: 0.72rem; padding: 10px 14px;
    border-radius: 4px; outline: none; min-width: 0; flex-shrink: 1;
  }
  button {
    background: #1a1a1a; color: #ffffff; border: none; padding: 10px 28px;
    font-family: 'Roboto Mono', monospace; font-size: 0.82rem; cursor: pointer;
    border-radius: 4px; font-weight: bold; letter-spacing: 0.05em;
  }
  button:hover { background: #333; }
  button:disabled { background: #ccc; color: #888; cursor: not-allowed; }
  #status { font-size: 0.78rem; color: #999; margin-top: 12px; min-height: 0; }
  #results { margin-top: 20px; }
  .summary { background: #f7f7f7; border: 1px solid #e8e8e8; border-radius: 4px; padding: 20px; margin-bottom: 20px; }
  .summary-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 12px; }
  .stat { text-align: center; }
  .stat-val { font-size: 1.8rem; color: #0f0f0f; }
  .stat-label { font-size: 0.7rem; color: #aaa; margin-top: 2px; }
  .filter-row { display: flex; gap: 8px; margin-bottom: 12px; }
  .filter-btn {
    background: #6b7280; color: #ffffff; border: none;
    font-size: 0.82rem; padding: 10px 28px; font-weight: normal;
    font-family: 'Roboto Mono', monospace; cursor: pointer; border-radius: 4px;
  }
  .filter-btn:hover { background: #4b5563; }
  .filter-btn.active { background: #1a1a1a; color: #fff; }
  table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
  th { text-align: left; color: #aaa; font-weight: normal; padding: 8px 10px; border-bottom: 1px solid #e8e8e8; letter-spacing: 0.06em; text-transform: uppercase; }
  td { padding: 10px 10px; border-bottom: 1px solid #f0f0f0; color: #444; word-break: break-all; }
  .amp { color: #059669; }
  .non { color: #dc2626; }
  .prob-cell { white-space: nowrap; min-width: 120px; }
  .prob-bar { display: inline-block; width: 56px; height: 6px; background: #efefef; border-radius: 3px; vertical-align: middle; margin-right: 6px; overflow: hidden; }
  .prob-fill { display: block; height: 100%; border-radius: 3px; }
  .prob-text { font-size: 0.75rem; color: #666; }
  .dl { margin-top: 16px; display: flex; gap: 8px; }
  .dl button { background: #059669; color: #ffffff; border: none; font-size: 0.82rem; padding: 10px 28px; font-weight: normal; }
  .dl button:hover { background: #047857; }
  .result-note { margin-top: 20px; font-size: 0.72rem; color: #999; border-left: 2px solid #e0e0e0; padding: 10px 14px; line-height: 1.7; text-align: justify; }
  .err { color: #dc2626; font-size: 0.8rem; }
  .example-btn { background: #2563eb; color: #ffffff; border: none; font-size: 0.82rem; padding: 10px 28px; font-weight: normal; white-space: nowrap; }
  .example-btn:hover { background: #1d4ed8; }
  .clear-btn { background: #dc2626; color: #ffffff; border: none; font-size: 0.82rem; padding: 10px 28px; font-weight: normal; white-space: nowrap; }
  .clear-btn:hover { background: #b91c1c; }
  .modal-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.45);
    z-index: 1000; align-items: center; justify-content: center;
  }
  .modal-overlay.open { display: flex; }
  .modal {
    background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
    padding: 32px; width: 100%; max-width: 480px; font-family: 'Roboto Mono', monospace;
    box-shadow: 0 8px 32px rgba(0,0,0,0.12);
  }
  .modal h2 { font-size: 1rem; font-weight: normal; letter-spacing: 0.08em; margin-bottom: 20px; }
  .modal label { font-size: 0.73rem; color: #999; text-transform: uppercase; letter-spacing: 0.07em; display: block; margin-bottom: 6px; margin-top: 16px; }
  .modal select, .modal textarea {
    width: 100%; background: #f7f7f7; border: 1px solid #e0e0e0;
    font-family: 'Roboto Mono', monospace; font-size: 0.82rem; padding: 10px 12px;
    border-radius: 4px; outline: none; color: #1a1a1a;
  }
  .modal textarea { height: 120px; resize: vertical; }
  .modal-actions { display: flex; gap: 10px; margin-top: 20px; justify-content: flex-end; }
  .modal-cancel { background: #f0f0f0; color: #555; border: none; font-size: 0.82rem; padding: 10px 22px; font-family: 'Roboto Mono', monospace; cursor: pointer; border-radius: 4px; }
  .modal-cancel:hover { background: #e0e0e0; }
  .modal-submit { background: #1a1a1a; color: #fff; border: none; font-size: 0.82rem; padding: 10px 22px; font-family: 'Roboto Mono', monospace; cursor: pointer; border-radius: 4px; font-weight: bold; }
  .modal-submit:hover { background: #333; }
  .feedback-link { color: #999; text-decoration: underline; cursor: pointer; background: none; border: none; font-family: inherit; font-size: inherit; font-weight: normal; padding: 0; }
  .feedback-link:hover { color: #333; background: none; }
  .logo-strip { margin-top: 20px; padding-top: 16px; border-top: 1px solid #f0f0f0; display: flex; flex-wrap: nowrap; align-items: flex-start; justify-content: center; gap: 40px; overflow-x: auto; }
  .logo-group { display: flex; flex-direction: column; align-items: center; gap: 10px; }
  .logo-group-label { font-size: 0.58rem; letter-spacing: 0.10em; text-transform: uppercase; color: #b0b8c8; }
  .logo-row { display: flex; align-items: flex-start; gap: 6px; }
  .logo-row img { width: auto; object-fit: contain; filter: grayscale(30%); opacity: 0.75; transition: opacity 0.2s, filter 0.2s; }
  .logo-row img:hover { opacity: 1; filter: grayscale(0%); }
</style>
</head>
<body>
<div class="wrap">
  <div class="title-row">
    <h1>AMPidentifier</h1>
    <span class="status-dot" id="statusDot" title="Checking server..."></span>
  </div>
  <p class="sub">A Python-based toolkit for predicting antimicrobial peptides using ensemble machine learning and physicochemical descriptors.</p>

  <div class="stats-section">
    <div class="stats-grid">
      <div class="stats-item">
        <span class="stats-val" id="statSeq">—</span>
        <span class="stats-lbl">sequences classified</span>
      </div>
      <div class="stats-item">
        <span class="stats-val" id="statRuns">—</span>
        <span class="stats-lbl">prediction runs</span>
      </div>
      <div class="stats-item">
        <span class="stats-val" id="statVisitors">—</span>
        <span class="stats-lbl">unique visitors</span>
      </div>
      <div class="stats-item">
        <span class="stats-val">3</span>
        <span class="stats-lbl">research groups using AMPidentifier as main tool</span>
      </div>
    </div>
  </div>

  <p class="notice">For advanced parameter control use the <a href="https://github.com/madsondeluna/AMPIdentifier" target="_blank">CLI version</a> or install via <a href="https://pypi.org/project/ampidentifier/" target="_blank">PyPI</a>: <code style="background:#f0f0f0;color:#444;padding:2px 8px;border-radius:4px;font-size:0.85em;">pip install ampidentifier</code></p>

  <div class="label-row">
    <label>FASTA sequences</label>
    <span class="seq-counter" id="seqCounter"></span>
  </div>
  <textarea id="fasta" placeholder=">SequenceID
KRIVQRIKDFLRNLVPRTES" oninput="updateCounter();validateFasta();"></textarea>
  <div id="validationErr" class="validation-err"></div>

  <div class="upload-row">
    <input type="file" id="fileInput" accept=".fasta,.fa,.txt" onchange="handleFileUpload(event)">
    <button class="upload-btn" onclick="document.getElementById('fileInput').click()">Upload .fasta</button>
  </div>

  <div class="row">
    <select id="model">
      <option value="voting">Voting Ensemble (RF + SVM + GB + XGB + LGBM)</option>
      <option value="rf">Random Forest</option>
      <option value="svm">SVM</option>
      <option value="gb">Gradient Boosting</option>
      <option value="xgb">XGBoost</option>
      <option value="lgbm">LightGBM</option>
    </select>
    <button id="runBtn" onclick="runPrediction()">Run</button>
    <button class="clear-btn" onclick="clearAll()">Clear</button>
    <button class="example-btn" onclick="loadExample()">Load example</button>
  </div>

  <div id="status"></div>
  <div id="results"></div>

  <footer>
    <p>Luna-Aragão, M. A., da Silva, R. L., Bezerra Neto, J. P., dos Santos-Silva, C. A., da Silva Santos, D. E. &amp; Benko&#8209;Iseppon, A. M. (2026).
    AMPidentifier: A Cross-Platform Ensemble Toolkit for Antimicrobial Peptide Prediction.
    GitHub repository: <a href="https://github.com/madsondeluna/AMPIdentifier" target="_blank">https://github.com/madsondeluna/AMPIdentifier</a></p>
    <p style="margin-top:8px;">This tool is officially registered with the <strong style="color:#555;">INPI &ndash; Instituto Nacional da Propriedade Industrial</strong> (Brazilian National Institute of Industrial Property), Registration No. <strong style="color:#555;">BR 51 2025 005859-4</strong>. It is a property of the <strong style="color:#555;">Universidade Federal de Pernambuco (UFPE)</strong> and the <strong style="color:#555;">Laboratório de Genética e Biotecnologia Vegetal (LGBV)</strong>.</p>
    <p style="margin-top:8px;">Developer: <a href="mailto:madsondeluna@gmail.com">madsondeluna@gmail.com</a> &nbsp;·&nbsp; <a href="https://madsondeluna.com" target="_blank">madsondeluna.com</a> &nbsp;·&nbsp; <button class="feedback-link" onclick="openFeedback()">Report issue / Suggest improvement</button> &nbsp;·&nbsp; <span style="color:#bbb;">v{{ version }}</span></p>

    <div class="logo-strip">
      <div class="logo-group">
        <div class="logo-group-label">Institutions</div>
        <div class="logo-row">
          <img src="/img/ufpe.png"     alt="Universidade Federal de Pernambuco"   style="height:36px;">
          <img src="/img/ufmg.png"     alt="Universidade Federal de Minas Gerais" style="height:34px;">
          <img src="/img/upe-logo.png" alt="Universidade de Pernambuco"           style="height:34px;">
        </div>
      </div>
      <div class="logo-group">
        <div class="logo-group-label">Departments</div>
        <div class="logo-row">
          <img src="/img/dqf.png"   alt="Departamento de Química Fundamental" style="height:36px;">
          <img src="/img/dgen.jpeg" alt="Departamento de Genética"            style="height:36px;">
        </div>
      </div>
      <div class="logo-group">
        <div class="logo-group-label">Funding</div>
        <div class="logo-row">
          <img src="/img/facepe.png"  alt="FACEPE"  style="height:38px;">
          <img src="/img/fapemig.png" alt="FAPEMIG" style="height:48px; margin: -3px -17px 0;">
        </div>
      </div>
      <div class="logo-group">
        <div class="logo-group-label">Research groups</div>
        <div class="logo-row">
          <img src="/img/lgbv.png" alt="Laboratório de Genética e Biotecnologia Vegetal" style="height:34px;">
          <img src="/img/lcm3.png" alt="LCM3"                                            style="height:34px;">
        </div>
      </div>
    </div>
  </footer>
</div>

<!-- Feedback modal -->
<div class="modal-overlay" id="feedbackOverlay" onclick="closeFeedbackOutside(event)">
  <div class="modal" role="dialog" aria-modal="true" aria-labelledby="feedbackTitle">
    <h2 id="feedbackTitle">Report issue / Suggest improvement</h2>
    <label for="feedbackType">Type</label>
    <select id="feedbackType">
      <option value="bug">Bug report</option>
      <option value="feature">Feature request</option>
      <option value="other">Other</option>
    </select>
    <label for="feedbackMsg">Description</label>
    <textarea id="feedbackMsg" placeholder="Describe the issue or your suggestion..."></textarea>
    <div class="modal-actions">
      <button class="modal-cancel" onclick="closeFeedback()">Cancel</button>
      <button class="modal-submit" onclick="submitFeedback()">Open on GitHub</button>
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

async function checkServerStatus() {
  const dot = document.getElementById('statusDot');
  try {
    const r = await fetch('/health', { cache: 'no-cache' });
    if (r.ok) { dot.classList.add('online');  dot.title = 'Server online'; }
    else       { dot.classList.add('offline'); dot.title = 'Server error';  }
  } catch(e) {
    dot.classList.add('offline'); dot.title = 'Server offline';
  }
}
checkServerStatus();

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

function updateCounter() {
  const n = (document.getElementById('fasta').value.match(/^>/gm) || []).length;
  document.getElementById('seqCounter').textContent =
    n > 0 ? n + ' sequence' + (n === 1 ? '' : 's') : '';
}

function validateFasta() {
  const text  = document.getElementById('fasta').value.trim();
  const errEl = document.getElementById('validationErr');
  if (!text) { errEl.textContent = ''; return true; }

  const lines = text.split('\\n').map(l => l.trim()).filter(Boolean);
  if (!lines[0].startsWith('>')) {
    errEl.textContent = 'Invalid format: first line must start with >.';
    return false;
  }

  let seq = '', headers = 0;
  for (const line of lines) {
    if (line.startsWith('>')) {
      if (seq) {
        if (seq.length < 5) { errEl.textContent = 'Sequence too short (min 5 residues).'; return false; }
        if (!VALID_AA.test(seq)) { errEl.textContent = 'Invalid characters in sequence.'; return false; }
      }
      seq = ''; headers++;
    } else {
      seq += line;
    }
  }
  if (seq) {
    if (seq.length < 5) { errEl.textContent = 'Sequence too short (min 5 residues).'; return false; }
    if (!VALID_AA.test(seq)) { errEl.textContent = 'Invalid characters in sequence.'; return false; }
  }
  if (!headers) { errEl.textContent = 'No valid FASTA sequences found.'; return false; }
  errEl.textContent = '';
  return true;
}

function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById('fasta').value = ev.target.result;
    updateCounter();
    document.getElementById('validationErr').textContent = '';
  };
  reader.readAsText(file);
}

function loadExample() {
  document.getElementById('fasta').value = EXAMPLE;
  updateCounter();
  document.getElementById('validationErr').textContent = '';
}

function clearAll() {
  document.getElementById('fasta').value = '';
  document.getElementById('results').innerHTML = '';
  document.getElementById('status').textContent = '';
  document.getElementById('seqCounter').textContent = '';
  document.getElementById('validationErr').textContent = '';
  document.getElementById('fileInput').value = '';
  lastData = null;
}

async function runPrediction() {
  const fasta  = document.getElementById('fasta').value.trim();
  const model  = document.getElementById('model').value;
  const btn    = document.getElementById('runBtn');
  const status = document.getElementById('status');
  const results = document.getElementById('results');

  if (!fasta) { status.innerHTML = '<span class="err">Paste at least one FASTA sequence.</span>'; return; }

  btn.disabled = true;
  status.textContent = 'Running prediction...';
  results.innerHTML  = '';

  const form = new FormData();
  form.append('fasta_sequence', fasta);
  form.append('model', model);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: form });
    const data = await res.json();
    if (data.error) {
      status.innerHTML = '<span class="err">Error: ' + data.error + '</span>';
    } else {
      lastData = data.predictions;
      status.textContent = '';
      renderResults(data);
      loadStats();
    }
  } catch (e) {
    status.innerHTML = '<span class="err">Request failed: ' + e.message + '</span>';
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
    const color  = isAmp ? '#059669' : '#dc2626';
    const fill   = prob !== null ? (prob * 100).toFixed(1) + '%' : '0%';
    const barHtml = prob !== null
      ? '<span class="prob-bar"><span class="prob-fill" style="width:' + fill + ';background:' + color + ';"></span></span><span class="prob-text">' + pct + '</span>'
      : '—';
    const label = isAmp
      ? '<span class="amp">AMP</span>'
      : '<span class="non">non-AMP</span>';
    return '<tr class="' + (isAmp ? 'r-amp' : 'r-non') + '"><td>' +
      (r.ID || r.id || '—') + '</td><td>' +
      (r.sequence || '—') + '</td><td>' +
      label + '</td><td class="prob-cell">' +
      barHtml + '</td></tr>';
  }

  const modelLabels = {
    voting: 'Voting Ensemble',
    rf: 'Random Forest',
    svm: 'SVM',
    gb: 'Gradient Boosting',
    xgb: 'XGBoost',
    lgbm: 'LightGBM'
  };

  document.getElementById('results').innerHTML =
    '<div class="summary">' +
      '<label>Results — ' + (modelLabels[data.model] || data.model) + '</label>' +
      '<div class="summary-grid">' +
        '<div class="stat"><div class="stat-val">' + total + '</div><div class="stat-label">sequences</div></div>' +
        '<div class="stat"><div class="stat-val" style="color:#059669">' + amps + '</div><div class="stat-label">predicted AMP</div></div>' +
        '<div class="stat"><div class="stat-val" style="color:#dc2626">' + (total - amps) + '</div><div class="stat-label">predicted non-AMP</div></div>' +
      '</div>' +
    '</div>' +
    '<div class="filter-row">' +
      '<button class="filter-btn active" id="fAll" onclick="applyFilter(\\'all\\')">All</button>' +
      '<button class="filter-btn" id="fAmp" onclick="applyFilter(\\'amp\\')">AMP only</button>' +
      '<button class="filter-btn" id="fNon" onclick="applyFilter(\\'non\\')">Non-AMP only</button>' +
    '</div>' +
    '<table id="tbl">' +
      '<thead><tr><th>ID</th><th>Sequence</th><th>Prediction</th><th>Prob. AMP</th></tr></thead>' +
      '<tbody>' + preds.map(makeRow).join('') + '</tbody>' +
    '</table>' +
    '<div class="dl">' +
      '<button onclick="downloadCSV()">Download CSV</button>' +
      '<button id="copyBtn" onclick="copyTable()">Copy table</button>' +
    '</div>' +
    '<div class="result-note">' +
      '<strong>Interpretation note:</strong> Predictions are computed from 22 physicochemical and compositional descriptors derived from the primary amino acid sequence. ' +
      'For higher predictive power, use <strong>Voting Ensemble mode</strong> (RF + SVM + GB + XGB + LGBM), which combines five independent classifiers by soft voting and achieves ' +
      '<strong>AUC-ROC 0.950</strong>, <strong>MCC 0.742</strong>, <strong>Sensitivity 94.9%</strong>, and <strong>Specificity 78.4%</strong> on the independent benchmark set (n = 4,736). ' +
      'Bear in mind that proteins whose primary function is not antimicrobial activity may still harbour potential antimicrobial features in specific sequence regions. A full benchmark comparison against 18 published tools is available in Luna-Arago et al. (2026), <a href="https://doi.org/10.1021/acs.jcim.XXXXXXX" target="_blank" style="color:#999;">doi:10.1021/acs.jcim.XXXXXXX</a>.' +
    '</div>';
}

function applyFilter(type) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  const ids = { all: 'fAll', amp: 'fAmp', non: 'fNon' };
  document.getElementById(ids[type]).classList.add('active');
  document.querySelectorAll('#tbl tbody tr').forEach(row => {
    if (type === 'all') row.style.display = '';
    else if (type === 'amp') row.style.display = row.classList.contains('r-amp') ? '' : 'none';
    else row.style.display = row.classList.contains('r-non') ? '' : 'none';
  });
}

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
  a.download = 'ampidentifier_' + new Date().toISOString().slice(0, 10) + '.csv';
  a.click();
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


@app.route('/')
def index():
    resp = make_response(render_template_string(PAGE, version=VERSION))
    if not request.cookies.get('_amp_sid'):
        resp.set_cookie('_amp_sid', str(uuid.uuid4()), max_age=365 * 24 * 3600, samesite='Lax', httponly=True)
    return resp


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/stats')
def stats():
    return jsonify(get_stats())


@app.route('/predict', methods=['POST'])
def predict():
    try:
        fasta_text   = request.form.get('fasta_sequence', '').strip()
        model_choice = request.form.get('model', 'voting')

        if not fasta_text:
            return jsonify({'error': 'No FASTA sequence provided'}), 400

        with tempfile.TemporaryDirectory() as tmp:
            fasta_path = os.path.join(tmp, 'input.fasta')
            with open(fasta_path, 'w') as f:
                f.write(fasta_text)

            sequences, seq_ids = load_fasta_sequences(fasta_path)
            if not sequences:
                return jsonify({'error': 'Invalid FASTA format or empty file'}), 400

            output_dir = os.path.join(tmp, 'out')
            os.makedirs(output_dir)

            with contextlib.redirect_stdout(io.StringIO()):
                run_prediction_pipeline(
                    input_file=fasta_path,
                    output_dir=output_dir,
                    internal_model_type=model_choice,
                    use_ensemble=False,
                )

            predictions_df = pd.read_csv(
                os.path.join(output_dir, f'predictions_{model_choice}.csv')
            )

            session_id = request.cookies.get('_amp_sid', str(uuid.uuid4()))
            try:
                increment_stats(len(sequences), session_id)
            except Exception:
                pass

            model_labels = {
                'voting': 'Voting Ensemble', 'rf': 'Random Forest',
                'svm': 'SVM', 'gb': 'Gradient Boosting',
                'xgb': 'XGBoost', 'lgbm': 'LightGBM',
            }
            _send_telegram(
                f'[AMPidentifier] New prediction run\n'
                f'Sequences: {len(sequences)}\n'
                f'Model: {model_labels.get(model_choice, model_choice)}\n'
                f'Session: {session_id[:8]}...'
            )

            return jsonify({
                'model': model_choice,
                'num_sequences': len(sequences),
                'predictions': predictions_df.to_dict(orient='records')
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
