"""
AMPidentifier Web Portal — v2.0
"""
import base64
import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import threading
import urllib.error
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
_USE_PG  = bool(os.environ.get('DATABASE_URL'))

if _USE_PG:
    _PH             = '%s'
    _INSERT_STAT    = 'INSERT INTO stats (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING'
    _INSERT_SESSION = 'INSERT INTO sessions (id) VALUES (%s) ON CONFLICT (id) DO NOTHING'
else:
    _PH             = '?'
    _INSERT_STAT    = 'INSERT OR IGNORE INTO stats VALUES (?, ?)'
    _INSERT_SESSION = 'INSERT OR IGNORE INTO sessions VALUES (?)'

def _db_path():
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stats.db')
    return os.environ.get('STATS_DB', default)

def _conn():
    if _USE_PG:
        import psycopg2
        return psycopg2.connect(os.environ['DATABASE_URL'], sslmode='require')
    c = sqlite3.connect(_db_path(), check_same_thread=False)
    c.execute('PRAGMA journal_mode=WAL')
    return c

def init_db():
    seed = {
        'total_sequences':  int(os.environ.get('STATS_SEQUENCES', 0)),
        'total_runs':       int(os.environ.get('STATS_RUNS', 0)),
        'unique_sessions':  int(os.environ.get('STATS_SESSIONS', 0)),
        'research_groups':  int(os.environ.get('STATS_RESEARCH_GROUPS', 3)),
    }
    with _db_lock:
        c = _conn()
        cur = c.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value INTEGER DEFAULT 0)')
        cur.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY)')
        for k, v in seed.items():
            cur.execute(_INSERT_STAT, (k, v))
        c.commit()
        cur.close()
        c.close()

def increment_stats(seq_count, session_id):
    with _db_lock:
        c = _conn()
        cur = c.cursor()
        cur.execute(f'UPDATE stats SET value = value + {_PH} WHERE key = {_PH}', (seq_count, 'total_sequences'))
        cur.execute(f'UPDATE stats SET value = value + 1 WHERE key = {_PH}', ('total_runs',))
        cur.execute(_INSERT_SESSION, (session_id,))
        if cur.rowcount > 0:
            cur.execute(f'UPDATE stats SET value = value + 1 WHERE key = {_PH}', ('unique_sessions',))
        c.commit()
        cur.close()
        c.close()

def get_stats():
    with _db_lock:
        c = _conn()
        cur = c.cursor()
        cur.execute('SELECT key, value FROM stats')
        rows = cur.fetchall()
        cur.close()
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
  @keyframes pulse-green {
    0%   { box-shadow: 0 0 0 0 rgba(5, 150, 105, 0.55); }
    70%  { box-shadow: 0 0 0 6px rgba(5, 150, 105, 0); }
    100% { box-shadow: 0 0 0 0 rgba(5, 150, 105, 0); }
  }
  .status-dot-wrapper { position: relative; display: inline-flex; align-items: center; flex-shrink: 0; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #ddd; transition: background 0.4s; cursor: default; }
  .status-dot.online  { background: #059669; animation: pulse-green 1.8s ease-out infinite; }
  .status-dot.offline { background: #dc2626; }
  .status-tooltip {
    display: none;
    position: absolute;
    left: 16px;
    top: 50%;
    transform: translateY(-50%);
    background: #1a1a1a;
    color: #e8e8e8;
    font-size: 0.68rem;
    line-height: 1.6;
    padding: 8px 12px;
    border-radius: 4px;
    white-space: nowrap;
    z-index: 100;
    pointer-events: none;
  }
  .status-tooltip::before {
    content: '';
    position: absolute;
    right: 100%;
    top: 50%;
    transform: translateY(-50%);
    border: 5px solid transparent;
    border-right-color: #1a1a1a;
  }
  .status-tooltip .tt-row { display: flex; align-items: center; gap: 7px; }
  .status-tooltip .tt-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
  .status-tooltip .tt-dot.c-green  { background: #059669; }
  .status-tooltip .tt-dot.c-red    { background: #dc2626; }
  .status-tooltip .tt-dot.c-gray   { background: #ddd; }
  .status-dot-wrapper:hover .status-tooltip { display: block; }
  .sub { font-size: 0.78rem; color: #888; margin-bottom: 10px; }
  .stats-section { margin-top: 12px; margin-bottom: 16px; text-align: center; }
  .stats-section-label { font-size: 0.65rem; color: #ccc; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 20px; }
  .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px 16px; align-items: start; justify-items: center; }
  @media (max-width: 720px) { .stats-grid { grid-template-columns: repeat(2, 1fr); } }
  .stats-item { display: flex; flex-direction: row; align-items: flex-start; gap: 10px; }
  .stats-val { font-size: 1.8rem; font-weight: 600; color: #1a1a1a; font-variant-numeric: tabular-nums; line-height: 1; flex-shrink: 0; }
  .stats-lbl { font-size: 0.62rem; color: #bbb; text-transform: uppercase; letter-spacing: 0.08em; text-align: left; line-height: 1.3; }
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
  .logo-strip { margin-top: 20px; padding-top: 16px; border-top: 1px solid #f0f0f0; display: flex; flex-wrap: wrap; align-items: flex-start; justify-content: center; gap: 24px 36px; overflow-x: visible; }
  .logo-group { display: flex; flex-direction: column; align-items: center; gap: 10px; }
  .logo-group-label { font-size: 0.58rem; letter-spacing: 0.10em; text-transform: uppercase; color: #b0b8c8; }
  .logo-row { display: flex; align-items: flex-start; gap: 6px; }
  .logo-row img { width: auto; object-fit: contain; filter: grayscale(30%); opacity: 0.75; transition: opacity 0.2s, filter 0.2s; }
  .logo-row img:hover { opacity: 1; filter: grayscale(0%); }
  .email-csv-section {
    margin-top: 20px; border: 1px solid #e0e0e0; border-radius: 4px;
    padding: 18px 20px; background: #f9f9f9;
  }
  .email-csv-header { margin-bottom: 4px; }
  .email-csv-title { font-size: 0.78rem; color: #444; font-weight: bold; letter-spacing: 0.04em; }
  .email-csv-desc { font-size: 0.72rem; color: #aaa; margin-top: 3px; }
  .email-csv-desc strong { color: #059669; font-weight: normal; }
  .email-csv-fields {
    display: grid; grid-template-columns: 1fr auto; gap: 10px;
    align-items: end; margin-top: 14px;
  }
  @media (max-width: 540px) {
    .email-csv-fields { grid-template-columns: 1fr; }
  }
  .email-csv-field label {
    font-size: 0.65rem; color: #bbb; text-transform: uppercase; letter-spacing: 0.07em;
    display: block; margin-bottom: 5px;
  }
  .email-csv-field input {
    width: 100%; background: #fff; border: 1px solid #e0e0e0; color: #1a1a1a;
    font-family: 'Roboto Mono', monospace; font-size: 0.8rem; padding: 9px 11px;
    border-radius: 4px; outline: none;
  }
  .email-csv-field input:focus { border-color: #aaa; }
  .email-csv-btn {
    background: #059669; color: #fff; border: none;
    font-family: 'Roboto Mono', monospace; font-size: 0.78rem; padding: 9px 20px;
    border-radius: 4px; cursor: pointer; font-weight: normal; white-space: nowrap;
    align-self: end;
  }
  .email-csv-btn:hover { background: #047857; }
  .email-csv-btn:disabled { background: #ccc; color: #888; cursor: not-allowed; }
  .email-csv-status { font-size: 0.72rem; margin-top: 10px; min-height: 16px; }
  .share-section {
    margin-top: 20px; margin-bottom: 20px;
    border: 1px solid #e0e0e0; border-left: 3px solid #1a1a1a;
    border-radius: 4px; padding: 16px 20px; background: #f9f9f9;
  }
  .share-inner { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px; }
  .share-heading { font-size: 0.82rem; color: #1a1a1a; font-weight: bold; letter-spacing: 0.03em; }
  .share-sub { font-size: 0.7rem; color: #aaa; margin-top: 3px; }
  .share-actions { display: flex; gap: 8px; flex-shrink: 0; }
  .share-btn {
    border: none; font-family: 'Roboto Mono', monospace; font-size: 0.75rem;
    padding: 8px 18px; border-radius: 4px; cursor: pointer; font-weight: normal;
    letter-spacing: 0.03em; white-space: nowrap;
  }
  .share-btn.copy-btn { background: #1a1a1a; color: #fff; }
  .share-btn.copy-btn:hover { background: #333; }
  .share-btn.gmail-btn { background: #fff; color: #1a1a1a; border: 1px solid #d0d0d0; }
  .share-btn.gmail-btn:hover { background: #f0f0f0; }
  .share-url-box {
    font-size: 0.72rem; color: #059669; background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 4px; padding: 7px 12px; margin-top: 12px; word-break: break-all; display: none;
  }
  .share-form { display: none; margin-top: 12px; gap: 8px; align-items: center; flex-wrap: wrap; }
  .share-form.open { display: flex; }
  .share-form input {
    flex: 1; min-width: 220px; padding: 8px 12px; border: 1px solid #d0d0d0;
    border-radius: 4px; font-family: 'Roboto Mono', monospace; font-size: 0.78rem;
    background: #fff; color: #1a1a1a;
  }
  .share-form input:focus { border-color: #1a1a1a; outline: none; }
  .share-form-status { font-size: 0.72rem; min-height: 16px; flex-basis: 100%; }
  .share-form-status .err { color: #dc2626; }
</style>
</head>
<body>
<div class="wrap">
  <div class="title-row">
    <h1>AMPidentifier</h1>
    <span class="status-dot-wrapper">
      <span class="status-dot" id="statusDot"></span>
      <span class="status-tooltip">
        <span class="tt-row"><span class="tt-dot c-green"></span> Server online</span>
        <span class="tt-row"><span class="tt-dot c-red"></span> Server offline or error</span>
        <span class="tt-row"><span class="tt-dot c-gray"></span> Checking connection...</span>
      </span>
    </span>
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
        <span class="stats-lbl">unique users</span>
      </div>
      <div class="stats-item wide">
        <span class="stats-val" id="statGroups">—</span>
        <span class="stats-lbl">research groups using as main tool for AMP prediction</span>
      </div>
    </div>
  </div>

  <div class="share-section">
    <div class="share-inner">
      <div class="share-text">
        <div class="share-heading">Find AMPidentifier useful?</div>
        <div class="share-sub">Share it with your lab or collaborators.</div>
      </div>
      <div class="share-actions">
        <button class="share-btn copy-btn" onclick="copyLink()" id="copyLinkBtn">Copy link</button>
        <button class="share-btn gmail-btn" onclick="toggleShareForm()" id="shareEmailBtn">Share by email</button>
      </div>
    </div>
    <div class="share-url-box" id="shareUrlBox"></div>
    <div class="share-form" id="shareForm">
      <input type="email" id="shareFriendEmail" placeholder="friend@example.com">
      <button class="share-btn copy-btn" onclick="sendShareEmail()" id="sendShareBtn">Send</button>
      <div class="share-form-status" id="shareFormStatus"></div>
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
    if (r.ok) { dot.classList.add('online'); }
    else       { dot.classList.add('offline'); }
  } catch(e) {
    dot.classList.add('offline');
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
    set('statGroups',   d.research_groups);
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
    '<div class="email-csv-section">' +
      '<div class="email-csv-header">' +
        '<div class="email-csv-title">Receive results by email</div>' +
        '<div class="email-csv-desc">A CSV with <strong>' + total + ' prediction' + (total !== 1 ? 's' : '') + '</strong> (' +
          '<strong style="color:#059669">' + amps + ' AMP</strong> / ' +
          '<strong style="color:#dc2626">' + (total - amps) + ' non-AMP</strong>) ' +
          'using <strong>' + (modelLabels[data.model] || data.model) + '</strong> will be sent to your inbox.' +
        '</div>' +
      '</div>' +
      '<div class="email-csv-fields">' +
        '<div class="email-csv-field">' +
          '<label for="csvEmail">Your email</label>' +
          '<input type="email" id="csvEmail" placeholder="you@example.com">' +
        '</div>' +
        '<button class="email-csv-btn" id="sendCsvBtn" onclick="sendCsvByEmail()">Send</button>' +
      '</div>' +
      '<div class="email-csv-status" id="emailCsvStatus"></div>' +
    '</div>' +
    '<div class="result-note">' +
      '<strong>Interpretation note:</strong> Predictions are computed from 22 physicochemical and compositional descriptors derived from the primary amino acid sequence. ' +
      'For higher predictive power, use <strong>Voting Ensemble mode</strong> (RF + SVM + GB + XGB + LGBM), which combines five independent classifiers by soft voting and achieves ' +
      '<strong>AUC-ROC 0.950</strong>, <strong>MCC 0.742</strong>, <strong>Sensitivity 94.9%</strong>, and <strong>Specificity 78.4%</strong> on the independent benchmark set (n = 4,736). ' +
      'Bear in mind that proteins whose primary function is not antimicrobial activity may still harbour potential antimicrobial features in specific sequence regions. A full benchmark is available in Luna-Arago et al. (2026), <a href="https://doi.org/10.1021/acs.jcim.XXXXXXX" target="_blank" style="color:#999;">doi:10.1021/acs.jcim.XXXXXXX</a>.' +
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
  if (!lastData) return;
  const email = document.getElementById('csvEmail').value.trim();
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
  btn.disabled = true;
  status.style.color = '#999';
  status.textContent = 'Sending...';
  try {
    const form = new FormData();
    form.append('to_email', email);
    form.append('csv_data', csv);
    const res = await fetch('/send_csv', { method: 'POST', body: form });
    const data = await res.json();
    if (data.ok) {
      status.style.color = '#059669';
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
  status.style.color = '#999';
  status.textContent = 'Sending...';
  try {
    const fd = new FormData();
    fd.append('to_email', email);
    const res = await fetch('/send_recommendation', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.ok) {
      status.style.color = '#059669';
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


@app.route('/send_csv', methods=['POST'])
def send_csv():
    api_key = os.environ.get('RESEND_API_KEY', '')
    if not api_key:
        return jsonify({'ok': False, 'error': 'Email service not configured.'}), 503

    to_email = request.form.get('to_email', '').strip()
    csv_data = request.form.get('csv_data', '').strip()

    if not to_email or '@' not in to_email:
        return jsonify({'ok': False, 'error': 'Invalid email address.'}), 400
    if not csv_data:
        return jsonify({'ok': False, 'error': 'No data to send.'}), 400

    from_addr = os.environ.get('RESEND_FROM_EMAIL', 'AMPidentifier <onboarding@resend.dev>')
    payload = json.dumps({
        'from': from_addr,
        'to': [to_email],
        'subject': '[AMPidentifier] Your prediction results',
        'text': (
            'Your AMPidentifier prediction results are attached as a CSV file.\n\n'
            '--\n'
            'AMPidentifier - Ensemble Machine Learning for AMP Prediction\n'
            'Luna-Aragao et al. (2026)\n'
        ),
        'attachments': [{
            'filename': 'ampidentifier_results.csv',
            'content': base64.b64encode(csv_data.encode('utf-8')).decode('ascii'),
        }],
    }).encode('utf-8')

    try:
        req = urllib.request.Request(
            'https://api.resend.com/emails',
            data=payload,
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req) as resp:
            resp.read()
        return jsonify({'ok': True})
    except urllib.error.HTTPError as e:
        raw = ''
        try:
            raw = e.read().decode('utf-8', errors='replace')
        except Exception:
            pass
        msg = ''
        try:
            data = json.loads(raw) if raw else {}
            msg = data.get('message') or data.get('error') or raw
        except Exception:
            msg = raw or str(e)
        app.logger.error(f'Resend HTTP {e.code}: {raw}')
        return jsonify({'ok': False, 'error': f'Resend {e.code}: {msg}'}), 500
    except Exception as e:
        app.logger.exception('send_csv failed')
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/send_recommendation', methods=['POST'])
def send_recommendation():
    api_key = os.environ.get('RESEND_API_KEY', '')
    if not api_key:
        return jsonify({'ok': False, 'error': 'Email service not configured.'}), 503

    to_email = request.form.get('to_email', '').strip()
    if not to_email or '@' not in to_email:
        return jsonify({'ok': False, 'error': 'Invalid email address.'}), 400

    from_addr = os.environ.get('RESEND_FROM_EMAIL', 'AMPidentifier <onboarding@resend.dev>')
    site_url = request.url_root.rstrip('/') + '/'

    subject = 'Someone recommends AMPidentifier to you'
    body = (
        'If you are receiving this message, it is because someone using AMPidentifier '
        'thought you might find it useful too.\n\n'
        'Hi! Hope you are doing well.\n\n'
        'AMPidentifier is a free tool for predicting antimicrobial peptides (AMPs) from FASTA sequences '
        'using an ensemble of five machine learning classifiers. It runs directly in the browser and requires no installation.\n\n'
        f'Check it out: {site_url}\n\n'
        '----------\n\n'
        'Si vous recevez ce message, c est parce que quelqu un utilisant AMPidentifier '
        'a pense que cet outil pourrait vous etre utile.\n\n'
        'Salut! Comment ca va?\n\n'
        'AMPidentifier est un outil gratuit pour predire les peptides antimicrobiens (AMP) a partir '
        'de sequences FASTA via un ensemble de cinq classificateurs. Il fonctionne directement dans le navigateur, sans installation.\n\n'
        f'Jette un coup d oeil: {site_url}\n\n'
        '----------\n\n'
        'Se voce esta recebendo esta mensagem, e porque alguem usando o AMPidentifier '
        'achou que poderia ser util para voce tambem.\n\n'
        'Ola! Tudo bem?\n\n'
        'O AMPidentifier e uma ferramenta gratuita para predizer peptideos antimicrobianos (AMPs) '
        'a partir de sequencias FASTA usando um ensemble de cinco modelos de machine learning. '
        'Roda direto no navegador e nao precisa instalar nada.\n\n'
        f'Da uma olhada: {site_url}\n\n'
        '----------\n'
        'Madson A. de Luna Aragao\n'
        'PhD Student in Bioinformatics @ UFMG | Belo Horizonte, Brazil\n'
        'madsondeluna@gmail.com | madsondeluna.com | delunalab.dev | linkedin.com/in/madsonaragao\n\n'
        'Reference: Luna-Aragao et al. (2026). AMPidentifier: A Cross-Platform Ensemble Toolkit for Antimicrobial Peptide Prediction.\n'
    )

    payload = json.dumps({
        'from': from_addr,
        'to': [to_email],
        'reply_to': 'madsondeluna@gmail.com',
        'subject': subject,
        'text': body,
    }).encode('utf-8')

    try:
        req = urllib.request.Request(
            'https://api.resend.com/emails',
            data=payload,
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req) as resp:
            resp.read()
        return jsonify({'ok': True})
    except urllib.error.HTTPError as e:
        raw = ''
        try:
            raw = e.read().decode('utf-8', errors='replace')
        except Exception:
            pass
        msg = ''
        try:
            data = json.loads(raw) if raw else {}
            msg = data.get('message') or data.get('error') or raw
        except Exception:
            msg = raw or str(e)
        app.logger.error(f'Resend HTTP {e.code}: {raw}')
        return jsonify({'ok': False, 'error': f'Resend {e.code}: {msg}'}), 500
    except Exception as e:
        app.logger.exception('send_recommendation failed')
        return jsonify({'ok': False, 'error': str(e)}), 500


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
            stats_now = get_stats()
            n_amp = int(predictions_df['prediction'].sum()) if 'prediction' in predictions_df.columns else 0
            avg_prob = predictions_df['probability_AMP'].mean() if 'probability_AMP' in predictions_df.columns else None
            prob_line = f'Avg AMP prob: {avg_prob:.3f}\n' if avg_prob is not None else ''
            _send_telegram(
                f'[AMPidentifier] New prediction run\n'
                f'Sequences: {len(sequences)} | AMPs: {n_amp} ({n_amp/len(sequences)*100:.0f}%)\n'
                f'{prob_line}'
                f'Model: {model_labels.get(model_choice, model_choice)}\n'
                f'Session: {session_id[:8]}...\n'
                f'\n'
                f'Total sequences classified: {stats_now.get("total_sequences", 0)}\n'
                f'Total prediction runs: {stats_now.get("total_runs", 0)}\n'
                f'Unique visitors: {stats_now.get("unique_sessions", 0)}'
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
