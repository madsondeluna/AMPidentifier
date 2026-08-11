"""
AMPidentifier Web Portal — v2.0
"""
import base64
import contextlib
import gzip
import io
import ipaddress
import json
import logging
import os
import sqlite3
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
import uuid

from flask import Flask, make_response, request, jsonify, render_template_string
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amp_identifier.core import run_prediction_pipeline
from amp_identifier.data_io import load_fasta_sequences

VERSION = "2.0.0"

def _post_telegram(text):
    """Synchronous Telegram send. No-op if not configured. Call from a worker thread."""
    token   = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        return
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

def _send_telegram(text):
    threading.Thread(target=lambda: _post_telegram(text), daemon=True).start()

_geo_log = logging.getLogger('ampidentifier.geo')
_geo_log.setLevel(logging.INFO)
if not _geo_log.handlers:
    _geo_handler = logging.StreamHandler()
    _geo_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    _geo_log.addHandler(_geo_handler)
_geo_log.propagate = False

def _client_ip():
    """Real client IP, honoring the proxy in front of the app.

    Order: CF-Connecting-IP (Cloudflare), X-Real-IP (Nginx), first of
    X-Forwarded-For (Render and most reverse proxies), then socket peer.
    """
    for h in ('CF-Connecting-IP', 'X-Real-IP'):
        v = request.headers.get(h, '')
        if v:
            return v.strip()
    fwd = request.headers.get('X-Forwarded-For', '')
    if fwd:
        return fwd.split(',')[0].strip()
    return request.remote_addr or ''

def _is_public_ip(ip):
    """True only for globally routable addresses. Local/private builds are filtered out."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified)

def _http_json(url, timeout=6):
    req = urllib.request.Request(
        url, headers={'User-Agent': 'AMPidentifier/2.0 (+https://www.ampidentifier.com)'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())

def _geo_ipapi_co(ip):
    data = _http_json(f'https://ipapi.co/{ip}/json/')
    if not data or data.get('error'):
        return None
    return {'city': data.get('city') or '', 'region': data.get('region') or '',
            'country': data.get('country_name') or '', 'country_code': data.get('country_code') or '',
            'lat': data.get('latitude'), 'lon': data.get('longitude')}

def _geo_ipwho_is(ip):
    data = _http_json(f'https://ipwho.is/{ip}')
    if not data or not data.get('success'):
        return None
    return {'city': data.get('city') or '', 'region': data.get('region') or '',
            'country': data.get('country') or '', 'country_code': data.get('country_code') or '',
            'lat': data.get('latitude'), 'lon': data.get('longitude')}

def _geo_ip_api_com(ip):
    # HTTP only on the free tier.
    data = _http_json(f'http://ip-api.com/json/{ip}')
    if not data or data.get('status') != 'success':
        return None
    return {'city': data.get('city') or '', 'region': data.get('regionName') or '',
            'country': data.get('country') or '', 'country_code': data.get('countryCode') or '',
            'lat': data.get('lat'), 'lon': data.get('lon')}

# ipapi.co rate-limits and blocks cloud egress IPs (Render/Fly/Railway); ipwho.is and
# ip-api.com are datacenter-friendly fallbacks. All free, no API key required.
_GEO_PROVIDERS = (
    ('ipapi.co',    _geo_ipapi_co),
    ('ipwho.is',    _geo_ipwho_is),
    ('ip-api.com',  _geo_ip_api_com),
)

def _geolocate(ip):
    """Resolve an IP to approximate location, trying providers in order. Returns dict or None.

    Each failure is logged (HTTP status when available) so the cause is never silently lost.
    """
    for name, fn in _GEO_PROVIDERS:
        try:
            result = fn(ip)
            if result and result.get('lat') is not None and result.get('lon') is not None:
                return result
            _geo_log.warning('geo provider %s returned no usable data', name)
        except urllib.error.HTTPError as e:
            _geo_log.warning('geo provider %s HTTP %s', name, e.code)
        except Exception as e:
            _geo_log.warning('geo provider %s failed: %s', name, e)
    _geo_log.warning('geo: all providers exhausted')
    return None

def record_usage_location(ip, message_template):
    """Fire-and-forget: resolve geo, store aggregate, send notification with location line.

    message_template carries a literal '{location}' placeholder marking where the
    'Location: ...' line should go. Notification is sent even if geo fails (without the line),
    and the placeholder is always cleared so it never reaches the user literally.
    """
    def _worker():
        location_line = ''
        if not ip:
            _geo_log.info('usage: empty client ip, skipping geo')
        elif not _is_public_ip(ip):
            _geo_log.info('usage: non-public client ip, skipping geo')
        else:
            geo = _geolocate(ip)
            if geo:
                city, region = geo['city'], geo['region']
                country, cc  = geo['country'], geo['country_code']
                try:
                    upsert_location(city, country, cc, geo['lat'], geo['lon'])
                except Exception as e:
                    _geo_log.warning('usage: location upsert failed: %s', e)
                parts = [p for p in (city, region, country) if p]
                if parts:
                    location_line = 'Location: ' + ', '.join(parts) + '\n'
        _post_telegram(message_template.replace('{location}', location_line))
    threading.Thread(target=_worker, daemon=True).start()

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
    }
    with _db_lock:
        c = _conn()
        cur = c.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value INTEGER DEFAULT 0)')
        cur.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY)')
        _coord = 'DOUBLE PRECISION' if _USE_PG else 'REAL'
        cur.execute(
            'CREATE TABLE IF NOT EXISTS locations ('
            'city_key TEXT PRIMARY KEY, '
            'city TEXT, '
            'country TEXT, '
            f'lat {_coord}, '
            f'lon {_coord}, '
            'count INTEGER DEFAULT 0)'
        )
        for k, v in seed.items():
            cur.execute(_INSERT_STAT, (k, v))
        c.commit()
        cur.close()
        c.close()

def increment_stats(seq_count, session_id):
    """Count the run and its sequences always; count the visitor only when identified.

    session_id is the browser's _amp_sid cookie. A request without it carries no
    identity: minting one here would turn every cookieless call, curl, API script,
    bot, into a fresh 'unique visitor'. The run still counts, the visitor does not.
    """
    with _db_lock:
        c = _conn()
        cur = c.cursor()
        cur.execute(f'UPDATE stats SET value = value + {_PH} WHERE key = {_PH}', (seq_count, 'total_sequences'))
        cur.execute(f'UPDATE stats SET value = value + 1 WHERE key = {_PH}', ('total_runs',))
        if session_id:
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

def upsert_location(city, country, country_code, lat, lon):
    city_key = f'{city}|{country_code}'
    with _db_lock:
        c = _conn()
        cur = c.cursor()
        if _USE_PG:
            cur.execute(
                'INSERT INTO locations (city_key, city, country, lat, lon, count) '
                'VALUES (%s, %s, %s, %s, %s, 1) '
                'ON CONFLICT (city_key) DO UPDATE SET count = locations.count + 1',
                (city_key, city, country, lat, lon))
        else:
            cur.execute('UPDATE locations SET count = count + 1 WHERE city_key = ?', (city_key,))
            if cur.rowcount == 0:
                cur.execute(
                    'INSERT OR IGNORE INTO locations (city_key, city, country, lat, lon, count) '
                    'VALUES (?, ?, ?, ?, ?, 1)',
                    (city_key, city, country, lat, lon))
        c.commit()
        cur.close()
        c.close()

def get_locations():
    with _db_lock:
        c = _conn()
        cur = c.cursor()
        cur.execute('SELECT city, country, lat, lon, count FROM locations ORDER BY count DESC')
        rows = cur.fetchall()
        cur.close()
        c.close()
    return [{'city': r[0], 'country': r[1], 'lat': r[2], 'lon': r[3], 'count': r[4]} for r in rows]

app = Flask(__name__, static_folder='img', static_url_path='/img')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
init_db()

ISSUES_URL = 'https://github.com/madsondeluna/AMPidentifier/issues'



EMAIL_FOOTER = {
    'en': (
        'This is an automated message, you do not need to reply.\n'
        'Your data is encrypted during transfer (HTTPS/TLS protocol) and never shared. We do not store your sequences.\n\n'
    ),
    'fr': (
        'Ceci est un message automatique, vous n\'avez pas besoin de répondre.\n'
        'Vos données sont chiffrées pendant le transfert (protocole HTTPS/TLS) et ne sont jamais partagées. Nous ne conservons pas vos séquences.\n\n'
    ),
    'es': (
        'Este es un mensaje automático, no es necesario responder.\n'
        'Tus datos se cifran durante la transferencia (protocolo HTTPS/TLS) y nunca se comparten. No almacenamos tus secuencias.\n\n'
    ),
    'pt': (
        'Esta é uma mensagem automática, você não precisa responder.\n'
        'Seus dados são criptografados durante a transferência (protocolo HTTPS/TLS) e nunca são compartilhados. Não armazenamos suas sequências.\n\n'
    ),
    'zh': (
        '这是一封自动邮件，您无需回复。\n'
        '您的数据在传输过程中加密 (HTTPS/TLS 协议)，且绝不共享。我们不会存储您的序列。\n\n'
    ),
}


def _wrap_email_html(text_body: str) -> str:
    import html as _html
    import re as _re
    escaped = _html.escape(text_body)
    escaped = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
    html_body = escaped.replace('\n', '<br>')
    return (
        '<!DOCTYPE html><html><body style="margin:0;padding:0;background:#ffffff;">'
        '<div style="max-width:680px;margin:0;padding:20px;background:#ffffff;text-align:left;'
        'font-family:Arial,Helvetica,sans-serif;font-size:14px;line-height:1.6;color:#222222;">'
        f'{html_body}'
        '</div></body></html>'
    )

MODEL_LABELS = {
    'voting': 'Voting Ensemble (RF + SVM + GB + XGB + LGBM)',
    'rf': 'Random Forest',
    'svm': 'SVM',
    'gb': 'Gradient Boosting',
    'xgb': 'XGBoost',
    'lgbm': 'LightGBM',
}

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AMPidentifier | Antimicrobial Peptide Prediction Tool</title>
<meta name="description" content="AMPidentifier is a free web tool for antimicrobial peptide (AMP) prediction using machine learning ensemble models. Submit FASTA sequences and classify AMPs in seconds.">
<meta name="keywords" content="antimicrobial peptide prediction, AMP classifier, machine learning peptides, bioinformatics tool, AMP identification, peptide analysis, ensemble model">
<meta name="author" content="Madson Aragao">
<link rel="canonical" href="https://www.ampidentifier.com/">
<meta property="og:type" content="website">
<meta property="og:url" content="https://www.ampidentifier.com/">
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
  "url": "https://www.ampidentifier.com/",
  "description": "AMPidentifier is an ensemble machine learning toolkit for antimicrobial peptide (AMP) prediction. It accepts FASTA sequences and returns AMP classification scores using gradient boosting, XGBoost, LightGBM, and a soft-voting ensemble model.",
  "applicationCategory": "Scientific Software",
  "operatingSystem": "Web",
  "offers": { "@type": "Offer", "price": "0", "priceCurrency": "USD" },
  "author": {
    "@type": "Person",
    "name": "Madson Aragao",
    "url": "https://github.com/madsondeluna"
  },
  "codeRepository": "https://github.com/madsondeluna/AMPidentifier",
  "license": "https://github.com/madsondeluna/AMPidentifier/blob/main/LICENSE",
  "keywords": ["antimicrobial peptide", "AMP prediction", "machine learning", "bioinformatics", "peptide classification"]
}
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:ital,wght@0,300;0,400;0,500;0,700;1,400&display=swap" rel="stylesheet">
<style>
  html { font-size: 17px; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Roboto Mono', monospace; background: #ffffff; color: #1a1a1a; min-height: 100vh; padding: 28px 24px; }
  .wrap { max-width: 760px; margin: 0 auto; }
  .title-row { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
  .status-row { display: flex; align-items: center; margin-bottom: 10px; font-size: 0.72rem; }
  h1 { font-size: 1.4rem; font-weight: normal; letter-spacing: 0.1em; color: #0f0f0f; }
  .brand-logo { height: 44px; width: auto; display: block; }
  @keyframes pulse-green {
    0%   { box-shadow: 0 0 0 0 rgba(5, 150, 105, 0.55); }
    70%  { box-shadow: 0 0 0 6px rgba(5, 150, 105, 0); }
    100% { box-shadow: 0 0 0 0 rgba(5, 150, 105, 0); }
  }
  .status-dot-wrapper { position: relative; display: inline-flex; align-items: center; flex-shrink: 0; gap: 6px; vertical-align: middle; }
  .status-label { font-size: inherit; color: #bbb; letter-spacing: 0.02em; transition: color 0.4s; }
  .status-label.online  { color: #059669; }
  .status-label.offline { color: #dc2626; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #ddd; transition: background 0.4s; cursor: default; }
  .status-dot.online  { background: #059669; animation: pulse-green 1.8s ease-out infinite; }
  .status-dot.offline { background: #dc2626; }
  .status-tooltip {
    display: none;
    position: absolute;
    left: calc(100% + 8px);
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
  .stats-section { margin-top: 8px; margin-bottom: 12px; text-align: center; }
  .stats-section-label { font-size: 0.65rem; color: #ccc; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 12px; }
  /* o primeiro numero comeca na borda esquerda do texto, o terceiro
     termina na direita e o do meio fica centrado: as tres colunas
     fecham na mesma largura do paragrafo acima. */
  .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px 10px; align-items: center; }
  .stats-item:nth-child(1) { justify-self: start; }
  .stats-item:nth-child(2) { justify-self: center; }
  .stats-item:nth-child(3) { justify-self: end; }
  @media (max-width: 720px) { .stats-grid { grid-template-columns: repeat(2, 1fr); } }
  .stats-item { display: flex; flex-direction: row; align-items: center; gap: 8px; }
  .stats-val { font-size: 1.8rem; font-weight: 600; color: #1a1a1a; font-variant-numeric: tabular-nums; line-height: 1; flex-shrink: 0; }
  .stats-lbl { font-size: 0.62rem; color: #bbb; text-transform: uppercase; letter-spacing: 0.08em; text-align: left; line-height: 1.3; }
  .notice { font-size: 0.75rem; color: #999; border-left: 2px solid #ddd; padding: 6px 12px; margin-bottom: 12px; line-height: 1.6; }
  .notice a { color: #555; text-decoration: underline; }
  .notice a:hover { color: #111; }
  .usage-map-section { margin-top: 28px; }
  .usage-map-label { font-size: 0.65rem; color: #ccc; font-weight: bold; letter-spacing: 0.12em; text-transform: uppercase; text-align: center; margin-bottom: 4px; }
  .usage-map-note { font-size: 0.62rem; color: #999; font-weight: bold; text-transform: uppercase; letter-spacing: 0.03em; text-align: center; line-height: 1.6; margin-bottom: 10px; }
  #usageMap { position: relative; width: 100%; border: 1px solid #e8e8e8; border-radius: 4px; background: #fff; padding: 8px 10px; }
  #usageMap svg { display: block; width: 100%; height: auto; }
  #usageMap .land { fill: #f6f6f6; stroke: #e2e2e2; stroke-width: 0.6; }
  #usageMap .ring { fill: #555555; fill-opacity: 0.28; stroke: #555555; stroke-width: 1.1; stroke-opacity: 0.9; transition: fill-opacity 0.15s ease, stroke-opacity 0.15s ease; }
  #usageMap .spot { cursor: default; }
  #usageMap .spot:hover .ring { fill-opacity: 0.45; stroke-opacity: 1; }
  .map-tip {
    position: absolute; pointer-events: none; opacity: 0; transform: translate(-50%, -100%);
    background: #ffffff; border: 1px solid #e0e0e0; border-radius: 4px; padding: 6px 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08); white-space: nowrap; transition: opacity 0.12s ease;
  }
  .map-tip .place { font-size: 0.7rem; color: #1a1a1a; }
  .map-tip .value { font-size: 0.62rem; color: #999; margin-top: 2px; }
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
  .validation-err { font-size: 0.73rem; color: #dc2626; margin-top: 4px; min-height: 0; }
  .validation-err:empty { margin-top: 0; }
  .upload-row { display: flex; align-items: center; gap: 8px; margin-top: 6px; }
  .upload-btn {
    background: #555555; color: #ffffff; border: none;
    font-size: 0.82rem; padding: 10px 28px; font-weight: normal;
    font-family: 'Roboto Mono', monospace; cursor: pointer; border-radius: 4px;
  }
  .upload-btn:hover { background: #444444; }
  #fileInput { display: none; }
  .row { display: flex; gap: 12px; margin-top: 8px; align-items: center; flex-wrap: nowrap; }
  select {
    background: #f7f7f7; border: 1px solid #e0e0e0; color: #1a1a1a;
    font-family: 'Roboto Mono', monospace; font-size: 0.72rem; padding: 10px 14px;
    border-radius: 4px; outline: none; min-width: 0; flex-shrink: 1;
    text-overflow: ellipsis;
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
    display: grid; grid-template-columns: auto 1fr auto; gap: 10px;
    align-items: end; margin-top: 14px;
  }
  @media (max-width: 540px) {
    .email-csv-fields { grid-template-columns: 1fr; }
  }
  .email-csv-field select {
    background: #fff; border: 1px solid #e0e0e0; color: #1a1a1a;
    font-family: 'Roboto Mono', monospace; font-size: 0.8rem; padding: 9px 28px 9px 11px;
    border-radius: 4px; outline: none; cursor: pointer; appearance: none;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'><path fill='%23666' d='M0 0l5 6 5-6z'/></svg>");
    background-repeat: no-repeat; background-position: right 10px center;
  }
  .email-csv-field select:focus { border-color: #aaa; }
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
    margin-top: 12px; margin-bottom: 14px;
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
  .share-form select {
    padding: 8px 28px 8px 10px; border: 1px solid #d0d0d0; border-radius: 4px;
    font-family: 'Roboto Mono', monospace; font-size: 0.78rem;
    background: #fff; color: #1a1a1a; cursor: pointer; appearance: none;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'><path fill='%23666' d='M0 0l5 6 5-6z'/></svg>");
    background-repeat: no-repeat; background-position: right 10px center;
  }
  .share-form select:focus { border-color: #1a1a1a; outline: none; }
  .share-form-status { font-size: 0.72rem; min-height: 16px; flex-basis: 100%; }
  .share-form-status .err { color: #dc2626; }
  .table-scroll { width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; }

  @media (max-width: 640px) {
    body { padding: 12px; }
    .wrap { max-width: 100%; }
    h1 { font-size: 1.4rem; }
    .brand-logo { height: 38px; }
    .stats-grid { grid-template-columns: repeat(3, 1fr); gap: 0 8px; align-items: start; }
    .stats-item { flex-direction: column; align-items: center; gap: 3px; }
    .stats-val { font-size: 1.15rem; }
    .stats-lbl { text-align: center; font-size: 0.55rem; letter-spacing: 0.05em; }
    .status-tooltip { display: none; }
    .share-inner { flex-direction: column; align-items: stretch; gap: 10px; }
    .share-actions { width: 100%; justify-content: stretch; }
    .share-actions .share-btn { flex: 1; padding: 10px 14px; }
    .share-form { flex-direction: column; align-items: stretch; }
    .share-form select, .share-form input, .share-form .share-btn { width: 100%; min-width: 0; }
    .email-csv-fields { grid-template-columns: 1fr !important; gap: 10px; }
    .email-csv-btn { width: 100%; padding: 12px 20px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .row select { grid-column: 1 / -1; width: 100%; }
    .row button { width: 100%; padding: 11px 8px; }
    .row .example-btn { grid-column: 1 / -1; }
    .upload-row { flex-wrap: wrap; }
    .upload-btn { width: 100%; }
    table { font-size: 0.72rem; min-width: 520px; }
    th, td { padding: 6px 8px !important; }
    td { word-break: normal; overflow-wrap: anywhere; }
    th:first-child, td:first-child { min-width: 150px; }
    th:nth-child(2), td:nth-child(2) { min-width: 120px; }
    .dl { flex-wrap: wrap; gap: 8px; }
    .dl button { flex: 1; min-width: 140px; }
    .filter-btn { padding: 6px 10px; font-size: 0.72rem; }
    .logo-strip { gap: 16px 20px; }
    footer { font-size: 0.6rem; text-align: left; }
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="title-row">
    <img src="/img/logo.png" alt="AMPidentifier" class="brand-logo">
  </div>
  <div class="status-row">
    <span class="status-dot-wrapper">
      <span class="status-dot" id="statusDot"></span>
      <span class="status-label" id="statusLabel">Checking</span>
      <span class="status-tooltip">
        <span class="tt-row"><span class="tt-dot c-green"></span> Online: model loaded, predictions ready</span>
        <span class="tt-row"><span class="tt-dot c-red"></span> Offline: backend unreachable, try again shortly</span>
        <span class="tt-row"><span class="tt-dot c-gray"></span> Checking server status...</span>
        <span class="tt-row" style="color:#999;margin-top:4px;font-size:0.62rem;">(Xms) = current /health round-trip latency</span>
      </span>
    </span>
  </div>
  <p class="sub"><strong>AMPidentifier</strong> is a toolkit for antimicrobial peptide prediction using ensemble machine learning (AUC 0.950, Sens. 0.949, Spec. 0.784). Free, no login required.</p>

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
      <select id="shareLang" title="Email language">
        <option value="en">EN</option>
        <option value="fr">FR</option>
        <option value="es">ES</option>
        <option value="pt">PT</option>
        <option value="zh">ZH</option>
      </select>
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

    <div class="usage-map-section">
      <div class="usage-map-label">Where AMPidentifier is being used</div>
      <div class="usage-map-note">Your data is encrypted during transfer (HTTPS/TLS protocol) and never shared. We do not store your sequences.</div>
      <div id="usageMap"></div>
    </div>

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

  Promise.all([
    fetch('/map-outline.json?v=1').then(function(r) { return r.json(); }),
    fetch('/locations', { cache: 'no-cache' }).then(function(r) { return r.json(); }),
  ]).then(function(res) {
    const world = res[0];
    let rows = res[1];
    if (!world || !Array.isArray(rows)) return;
    rows = rows.filter(function(d) { return d.lat != null && d.lon != null; });

    const scale = world.w / 360, latTop = 84;
    const px = function(lon) { return (lon + 180) * scale; };
    const py = function(lat) { return (latTop - lat) * scale; };

    const svg = el('svg', {
      viewBox: '0 0 ' + world.w + ' ' + world.h,
      role: 'img',
      'aria-label': 'World map of AMPidentifier usage',
    }, host);
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
      const floor = 3.5 / unit;
      rings.forEach(function(item) {
        item.node.setAttribute('r', Math.max(item.r, floor));
      });
    };

    rows.slice().sort(function(a, b) { return a.count - b.count; }).forEach(function(d) {
      const x = px(d.lon), y = py(d.lat);
      const r = 4 + 14 * Math.sqrt(d.count / max);
      const place = d.city ? (d.city + ', ' + d.country) : (d.country || 'Unknown');
      const p = total > 0 ? (d.count / total * 100) : 0;
      const pct = p < 1 ? '<1%' : Math.round(p) + '%';
      const value = pct + ' of all predictions';
      const g = el('g', { class: 'spot', tabindex: '0', role: 'img',
                          'aria-label': place + ', ' + value }, svg);
      const ring = el('circle', { cx: x, cy: y, r: r, class: 'ring' }, g);
      rings.push({ node: ring, r: r });
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
  }).catch(function() {});
}
initUsageMap();

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
  lastModel = null;
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
      lastModel = data.model;
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
    '<div class="table-scroll">' +
      '<table id="tbl">' +
        '<thead><tr><th>ID</th><th>Sequence</th><th>Prediction</th><th>Prob. AMP</th></tr></thead>' +
        '<tbody>' + preds.map(makeRow).join('') + '</tbody>' +
      '</table>' +
    '</div>' +
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
          '<label for="csvEmailLang">Lang</label>' +
          '<select id="csvEmailLang">' +
            '<option value="en">EN</option>' +
            '<option value="fr">FR</option>' +
            '<option value="es">ES</option>' +
            '<option value="pt">PT</option>' +
            '<option value="zh">ZH</option>' +
          '</select>' +
        '</div>' +
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
      '<strong>AUC-ROC 0.950</strong>, <strong>MCC 0.742</strong>, <strong>Sensitivity 94.9%</strong>, and <strong>Specificity 78.4%</strong> on the independent benchmark set. ' +
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
  if (!lastData || !lastModel) return;
  const email = document.getElementById('csvEmail').value.trim();
  const lang  = document.getElementById('csvEmailLang').value;
  const status = document.getElementById('emailCsvStatus');
  const btn = document.getElementById('sendCsvBtn');
  if (!email || !/^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(email)) {
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
  status.style.color = '#999';
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
  if (!email || !/^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(email)) {
    status.innerHTML = '<span class="err">Enter a valid email.</span>';
    return;
  }
  btn.disabled = true;
  status.style.color = '#999';
  status.textContent = 'Sending...';
  try {
    const fd = new FormData();
    fd.append('to_email', email);
    fd.append('lang', document.getElementById('shareLang').value);
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


@app.route('/google2a0f51da71f41d93.html')
def google_verify():
    return make_response(
        'google-site-verification: google2a0f51da71f41d93.html',
        200,
        {'Content-Type': 'text/html'}
    )


@app.route('/robots.txt')
def robots():
    content = (
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /predict\n"
        "Disallow: /send_csv\n"
        "Disallow: /send_recommendation\n"
        "Sitemap: https://www.ampidentifier.com/sitemap.xml\n"
    )
    return make_response(content, 200, {'Content-Type': 'text/plain'})


@app.route('/sitemap.xml')
def sitemap():
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        '  <url>\n'
        '    <loc>https://www.ampidentifier.com/</loc>\n'
        '    <changefreq>monthly</changefreq>\n'
        '    <priority>1.0</priority>\n'
        '  </url>\n'
        '</urlset>\n'
    )
    return make_response(content, 200, {'Content-Type': 'application/xml'})


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/stats')
def stats():
    return jsonify(get_stats())


@app.route('/locations')
def locations():
    return jsonify(get_locations())


_MAP_OUTLINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'world-outline.json')
_map_outline_gz = None

@app.route('/map-outline.json')
def map_outline():
    global _map_outline_gz
    if _map_outline_gz is None:
        with open(_MAP_OUTLINE_PATH, 'rb') as fh:
            _map_outline_gz = gzip.compress(fh.read(), 9)
    resp = make_response(_map_outline_gz)
    resp.headers['Content-Type']     = 'application/json'
    resp.headers['Content-Encoding'] = 'gzip'
    resp.headers['Cache-Control']    = 'public, max-age=31536000, immutable'
    return resp


@app.route('/send_csv', methods=['POST'])
def send_csv():
    api_key = os.environ.get('RESEND_API_KEY', '')
    if not api_key:
        return jsonify({'ok': False, 'error': 'Email service not configured.'}), 503

    to_email = request.form.get('to_email', '').strip()
    csv_data = request.form.get('csv_data', '').strip()
    lang     = request.form.get('lang', 'en').strip().lower()
    model    = request.form.get('model', 'voting').strip().lower()
    try:
        total = int(request.form.get('total', '0'))
        amps  = int(request.form.get('amps',  '0'))
    except ValueError:
        total, amps = 0, 0
    non_amps = max(total - amps, 0)
    if lang not in ('en', 'fr', 'es', 'pt', 'zh'):
        lang = 'en'

    if not to_email or '@' not in to_email:
        return jsonify({'ok': False, 'error': 'Invalid email address.'}), 400
    if not csv_data:
        return jsonify({'ok': False, 'error': 'No data to send.'}), 400

    from_addr = os.environ.get('RESEND_FROM_EMAIL', 'AMPidentifier <onboarding@resend.dev>')
    site_url = request.url_root.rstrip('/') + '/'
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    model_label = MODEL_LABELS.get(model, model)

    messages = {
        'en': {
            'subject': '[AMPidentifier] Your prediction results',
            'body': (
                'Dear user,\n\n'
                'Your AMPidentifier analysis has been completed. The CSV file containing the full '
                'results is attached to this message.\n\n'
                'Analysis summary:\n'
                f'Date/time: **{timestamp}**\n'
                f'Model used: **{model_label}**\n'
                f'Total: **{total} sequence(s)**\n'
                f'Predicted AMP: **{amps}**\n'
                f'Non-AMP: **{non_amps}**\n\n'
                'Interpretation note:\n'
                'Predictions are computed from 22 physicochemical and compositional descriptors '
                'derived from the primary amino acid sequence. For higher predictive power, the '
                'Voting Ensemble mode (RF + SVM + GB + XGB + LGBM) combines five independent '
                'classifiers by soft voting and achieves AUC-ROC 0.950, MCC 0.742, Sensitivity 94.9%, '
                'and Specificity 78.4% on the independent benchmark set. Please note '
                'that proteins whose primary function is not antimicrobial activity may still '
                'harbour potential antimicrobial features in specific sequence regions. The full '
                'benchmark is available in Luna-Aragão et al. (2026).\n\n'
                f'To run another analysis, please visit: {site_url}\n\n'
                'Documentation and source code: https://github.com/madsondeluna/AMPidentifier\n\n'
                'For questions, suggestions, or feedback: ampidentifier@delunalab.dev\n\n'
            ),
        },
        'fr': {
            'subject': '[AMPidentifier] Vos résultats de prédiction',
            'body': (
                'Madame, Monsieur,\n\n'
                'Votre analyse AMPidentifier a été menée à terme. Le fichier CSV contenant '
                'l\'ensemble des résultats est joint à ce message.\n\n'
                'Résumé de l\'analyse:\n'
                f'Date/heure: **{timestamp}**\n'
                f'Modèle utilisé: **{model_label}**\n'
                f'Total: **{total} séquence(s)**\n'
                f'AMP prédit: **{amps}**\n'
                f'Non-AMP: **{non_amps}**\n\n'
                'Note d\'interprétation:\n'
                'Les prédictions sont calculées à partir de 22 descripteurs physico-chimiques et '
                'compositionnels dérivés de la séquence primaire d\'acides aminés. Pour une meilleure '
                'puissance prédictive, le mode Voting Ensemble (RF + SVM + GB + XGB + LGBM) combine '
                'cinq classificateurs indépendants par soft voting et atteint AUC-ROC 0.950, MCC 0.742, '
                'Sensibilité 94.9% et Spécificité 78.4% sur le benchmark indépendant. '
                'Veuillez noter que des protéines dont la fonction principale n\'est pas l\'activité '
                'antimicrobienne peuvent néanmoins présenter des caractéristiques antimicrobiennes '
                'potentielles dans des régions spécifiques de la séquence. Le benchmark complet est '
                'disponible dans Luna-Aragão et al. (2026).\n\n'
                f'Pour effectuer une nouvelle analyse, veuillez consulter: {site_url}\n\n'
                'Documentation et code source: https://github.com/madsondeluna/AMPidentifier\n\n'
                'Pour toute question, suggestion ou retour: ampidentifier@delunalab.dev\n\n'
            ),
        },
        'es': {
            'subject': '[AMPidentifier] Sus resultados de predicción',
            'body': (
                'Estimado/a usuario/a,\n\n'
                'Su análisis en AMPidentifier ha sido completado. El archivo CSV con los resultados '
                'completos se encuentra adjunto a este mensaje.\n\n'
                'Resumen del análisis:\n'
                f'Fecha/hora: **{timestamp}**\n'
                f'Modelo usado: **{model_label}**\n'
                f'Total: **{total} secuencia(s)**\n'
                f'AMP predicho: **{amps}**\n'
                f'No-AMP: **{non_amps}**\n\n'
                'Nota de interpretación:\n'
                'Las predicciones se calculan a partir de 22 descriptores fisicoquímicos y composicionales '
                'derivados de la secuencia primaria de aminoácidos. Para mayor poder predictivo, el modo '
                'Voting Ensemble (RF + SVM + GB + XGB + LGBM) combina cinco clasificadores independientes '
                'mediante soft voting y alcanza AUC-ROC 0.950, MCC 0.742, Sensibilidad 94.9% y '
                'Especificidad 78.4% en el benchmark independiente. Tenga en cuenta que '
                'las proteínas cuya función principal no es la actividad antimicrobiana pueden aún '
                'presentar características antimicrobianas potenciales en regiones específicas de la '
                'secuencia. El benchmark completo está disponible en Luna-Aragão et al. (2026).\n\n'
                f'Para realizar un nuevo análisis, visite: {site_url}\n\n'
                'Documentación y código fuente: https://github.com/madsondeluna/AMPidentifier\n\n'
                'Para preguntas, sugerencias o comentarios: ampidentifier@delunalab.dev\n\n'
            ),
        },
        'pt': {
            'subject': '[AMPidentifier] Seus resultados de predição',
            'body': (
                'Prezado(a) usuário(a),\n\n'
                'Sua análise no AMPidentifier foi concluída. O arquivo CSV contendo os resultados '
                'completos encontra-se anexado a esta mensagem.\n\n'
                'Resumo da análise:\n'
                f'Data/hora: **{timestamp}**\n'
                f'Modelo: **{model_label}**\n'
                f'Total: **{total} sequência(s)**\n'
                f'AMP previsto: **{amps}**\n'
                f'Não-AMP: **{non_amps}**\n\n'
                'Nota de interpretação:\n'
                'As predições são computadas a partir de 22 descritores físico-químicos e composicionais '
                'derivados da sequência primária de aminoácidos. Para maior poder preditivo, o modo '
                'Voting Ensemble (RF + SVM + GB + XGB + LGBM) combina cinco classificadores independentes '
                'por soft voting e atinge AUC-ROC 0,950, MCC 0,742, Sensibilidade 94,9% e Especificidade '
                '78,4% no conjunto benchmark independente. Note que proteínas cuja função '
                'primária não é a atividade antimicrobiana podem, ainda assim, apresentar '
                'características antimicrobianas potenciais em regiões específicas da sequência. '
                'O benchmark completo está disponível em Luna-Aragão et al. (2026).\n\n'
                f'Para realizar uma nova análise, acesse: {site_url}\n\n'
                'Documentação e código-fonte: https://github.com/madsondeluna/AMPidentifier\n\n'
                'Para dúvidas, sugestões ou comentários: ampidentifier@delunalab.dev\n\n'
            ),
        },
        'zh': {
            'subject': '[AMPidentifier] 您的预测结果',
            'body': (
                '尊敬的用户，您好。\n\n'
                '您在 AMPidentifier 上的分析已完成。包含完整结果的 CSV 文件已作为附件随本邮件一同发送。\n\n'
                '分析摘要：\n'
                f'日期/时间: **{timestamp}**\n'
                f'使用模型: **{model_label}**\n'
                f'总数: **{total} 个序列**\n'
                f'预测为 AMP: **{amps}**\n'
                f'非 AMP: **{non_amps}**\n\n'
                '解释说明：\n'
                '预测基于从氨基酸主序列衍生的 22 个理化和组成描述符计算。为获得更高的预测能力，'
                'Voting Ensemble 模式（RF + SVM + GB + XGB + LGBM）通过 soft voting 组合五个独立分类器，'
                '在独立基准集上达到 AUC-ROC 0.950、MCC 0.742、敏感度 94.9%、特异度 78.4%。'
                '敬请注意，主要功能并非抗菌活性的蛋白质仍可能在特定序列区域具有潜在的抗菌特征。'
                '完整基准结果详见 Luna-Aragão et al. (2026)。\n\n'
                f'如需进行新的分析，请访问：{site_url}\n\n'
                '文档与源代码：https://github.com/madsondeluna/AMPidentifier\n\n'
                '如有问题、建议或反馈，请发送邮件至：ampidentifier@delunalab.dev\n\n'
            ),
        },
    }

    subject = messages[lang]['subject']
    body = messages[lang]['body'] + EMAIL_FOOTER[lang]

    payload = json.dumps({
        'from': from_addr,
        'to': [to_email],
        'reply_to': 'madsondeluna@gmail.com',
        'subject': subject,
        'text': body,
        'html': _wrap_email_html(body),
        'attachments': [{
            'filename': 'ampidentifier_results.csv',
            'content': base64.b64encode(csv_data.encode('utf-8')).decode('ascii'),
        }],
    }).encode('utf-8')

    try:
        req = urllib.request.Request(
            'https://api.resend.com/emails',
            data=payload,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'AMPidentifier/2.0 (https://ampidentifier.com)',
                'Accept': 'application/json',
            },
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
    lang = request.form.get('lang', 'en').strip().lower()
    if lang not in ('en', 'fr', 'es', 'pt', 'zh'):
        lang = 'en'
    if not to_email or '@' not in to_email:
        return jsonify({'ok': False, 'error': 'Invalid email address.'}), 400

    from_addr = os.environ.get('RESEND_FROM_EMAIL', 'AMPidentifier <onboarding@resend.dev>')
    site_url = request.url_root.rstrip('/') + '/'

    issues_url = ISSUES_URL

    metrics_block = (
        'AUC-ROC: **0.950**\n'
        'MCC: **0.742**\n'
        'Sensitivity: **94.9%**\n'
        'Specificity: **78.4%**\n'
    )

    messages = {
        'en': {
            'subject': 'AMPidentifier has been recommended to you',
            'body': (
                'Dear researcher,\n\n'
                'This message has been sent to you because a user of AMPidentifier considered that '
                'this tool may be of interest to your work.\n\n'
                'AMPidentifier is a free, open-source tool for predicting antimicrobial peptides (AMPs) '
                'from FASTA sequences. It uses a Voting Ensemble of five machine learning classifiers '
                '(Random Forest, SVM, Gradient Boosting, XGBoost, LightGBM) trained on 22 physicochemical '
                'and compositional descriptors derived from the primary amino acid sequence.\n\n'
                'Official metrics:\n'
                f'{metrics_block}\n'
                'Built with Python, scikit-learn, XGBoost, LightGBM, Flask, and PostgreSQL. '
                'Runs directly in the browser, no installation required.\n\n'
                'Available in three formats:\n'
                f'Web: {site_url}\n'
                'CLI / repo: https://github.com/madsondeluna/AMPidentifier\n'
                'Python package: pip install ampidentifier (https://pypi.org/project/ampidentifier/)\n\n'
                'For questions, suggestions, or feedback: ampidentifier@delunalab.dev\n\n'
            ),
        },
        'fr': {
            'subject': 'AMPidentifier vous a été recommandé',
            'body': (
                'Madame, Monsieur,\n\n'
                'Ce message vous a été envoyé parce qu\'un utilisateur d\'AMPidentifier a estimé que '
                'cet outil pourrait être utile à vos travaux.\n\n'
                'AMPidentifier est un outil gratuit et open-source pour prédire les peptides '
                'antimicrobiens (AMP) à partir de séquences FASTA. Il utilise un Voting Ensemble '
                'de cinq classificateurs de machine learning (Random Forest, SVM, Gradient Boosting, '
                'XGBoost, LightGBM) entraînés sur 22 descripteurs physico-chimiques et compositionnels '
                'dérivés de la séquence primaire d\'acides aminés.\n\n'
                'Métriques officielles:\n'
                'AUC-ROC: **0.950**\n'
                'MCC: **0.742**\n'
                'Sensibilité: **94.9%**\n'
                'Spécificité: **78.4%**\n\n'
                'Construit avec Python, scikit-learn, XGBoost, LightGBM, Flask et PostgreSQL. '
                'Fonctionne directement dans le navigateur, sans installation.\n\n'
                'Disponible en trois formats:\n'
                f'Web: {site_url}\n'
                'CLI / dépôt: https://github.com/madsondeluna/AMPidentifier\n'
                'Paquet pip: pip install ampidentifier (https://pypi.org/project/ampidentifier/)\n\n'
                'Pour toute question, suggestion ou retour: ampidentifier@delunalab.dev\n\n'
            ),
        },
        'es': {
            'subject': 'AMPidentifier le ha sido recomendado',
            'body': (
                'Estimado/a colega,\n\n'
                'Este mensaje le ha sido enviado porque un usuario de AMPidentifier consideró '
                'que esta herramienta podría ser de utilidad para su trabajo.\n\n'
                'AMPidentifier es una herramienta gratuita y de código abierto para predecir '
                'péptidos antimicrobianos (AMPs) a partir de secuencias FASTA. Utiliza un Voting Ensemble '
                'de cinco clasificadores de machine learning (Random Forest, SVM, Gradient Boosting, '
                'XGBoost, LightGBM) entrenados con 22 descriptores fisicoquímicos y composicionales '
                'derivados de la secuencia primaria de aminoácidos.\n\n'
                'Métricas oficiales:\n'
                'AUC-ROC: **0.950**\n'
                'MCC: **0.742**\n'
                'Sensibilidad: **94.9%**\n'
                'Especificidad: **78.4%**\n\n'
                'Construido con Python, scikit-learn, XGBoost, LightGBM, Flask y PostgreSQL. '
                'Funciona directamente en el navegador, sin instalación.\n\n'
                'Disponible en tres formatos:\n'
                f'Web: {site_url}\n'
                'CLI / repo: https://github.com/madsondeluna/AMPidentifier\n'
                'Paquete pip: pip install ampidentifier (https://pypi.org/project/ampidentifier/)\n\n'
                'Para preguntas, sugerencias o comentarios: ampidentifier@delunalab.dev\n\n'
            ),
        },
        'pt': {
            'subject': 'O AMPidentifier foi recomendado a você',
            'body': (
                'Prezado(a) colega,\n\n'
                'Esta mensagem foi enviada a você porque um usuário do AMPidentifier considerou '
                'que esta ferramenta pode ser de interesse para o seu trabalho.\n\n'
                'O AMPidentifier é uma ferramenta gratuita e open-source para predizer peptídeos '
                'antimicrobianos (AMPs) a partir de sequências FASTA. Utiliza um Voting Ensemble de cinco '
                'classificadores de machine learning (Random Forest, SVM, Gradient Boosting, XGBoost, '
                'LightGBM) treinados com 22 descritores físico-químicos e composicionais derivados da '
                'sequência primária de aminoácidos.\n\n'
                'Métricas oficiais:\n'
                'AUC-ROC: **0.950**\n'
                'MCC: **0.742**\n'
                'Sensibilidade: **94.9%**\n'
                'Especificidade: **78.4%**\n\n'
                'Construído com Python, scikit-learn, XGBoost, LightGBM, Flask e PostgreSQL. '
                'Executa diretamente no navegador, sem necessidade de instalação.\n\n'
                'Disponível em três formatos:\n'
                f'Web: {site_url}\n'
                'CLI / repo: https://github.com/madsondeluna/AMPidentifier\n'
                'Pacote pip: pip install ampidentifier (https://pypi.org/project/ampidentifier/)\n\n'
                'Para dúvidas, sugestões ou comentários: ampidentifier@delunalab.dev\n\n'
            ),
        },
        'zh': {
            'subject': '向您推荐 AMPidentifier',
            'body': (
                '尊敬的同行，您好。\n\n'
                '本邮件之所以发送给您，是因为一位 AMPidentifier 用户认为该工具可能对您的工作有所助益。\n\n'
                'AMPidentifier 是一款免费的开源工具，用于从 FASTA 序列预测抗菌肽（AMPs）。'
                '它采用五个机器学习分类器（Random Forest、SVM、Gradient Boosting、XGBoost、LightGBM）的'
                ' Voting Ensemble，基于从氨基酸主序列衍生的 22 个理化和组成描述符进行训练。\n\n'
                '官方指标：\n'
                'AUC-ROC: **0.950**\n'
                'MCC: **0.742**\n'
                '敏感度（Sensitivity）: **94.9%**\n'
                '特异度（Specificity）: **78.4%**\n\n'
                '本工具基于 Python、scikit-learn、XGBoost、LightGBM、Flask 与 PostgreSQL 构建，'
                '可直接在浏览器中运行，无需安装。\n\n'
                '提供以下三种使用方式：\n'
                f'Web: {site_url}\n'
                'CLI / repo: https://github.com/madsondeluna/AMPidentifier\n'
                'Python 包: pip install ampidentifier (https://pypi.org/project/ampidentifier/)\n\n'
                '如有问题、建议或反馈，请发送邮件至：ampidentifier@delunalab.dev\n\n'
            ),
        },
    }

    subject = messages[lang]['subject']
    body = messages[lang]['body'] + EMAIL_FOOTER[lang]

    payload = json.dumps({
        'from': from_addr,
        'to': [to_email],
        'reply_to': 'madsondeluna@gmail.com',
        'subject': subject,
        'text': body,
        'html': _wrap_email_html(body),
    }).encode('utf-8')

    try:
        req = urllib.request.Request(
            'https://api.resend.com/emails',
            data=payload,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'AMPidentifier/2.0 (https://ampidentifier.com)',
                'Accept': 'application/json',
            },
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

            n_amp = int(predictions_df['prediction'].sum()) if 'prediction' in predictions_df.columns else 0
            session_id = request.cookies.get('_amp_sid', '')
            try:
                increment_stats(len(sequences), session_id)
            except Exception as e:
                _geo_log.warning('stats: increment failed: %s', e)

            model_labels = {
                'voting': 'Voting Ensemble', 'rf': 'Random Forest',
                'svm': 'SVM', 'gb': 'Gradient Boosting',
                'xgb': 'XGBoost', 'lgbm': 'LightGBM',
            }
            stats_now = get_stats()
            avg_prob = predictions_df['probability_AMP'].mean() if 'probability_AMP' in predictions_df.columns else None
            prob_line = f'Avg AMP prob: {avg_prob:.3f}\n' if avg_prob is not None else ''
            message_template = (
                f'[AMPidentifier] New prediction run\n'
                f'Sequences: {len(sequences)} | AMPs: {n_amp} ({n_amp/len(sequences)*100:.0f}%)\n'
                f'{prob_line}'
                f'Model: {model_labels.get(model_choice, model_choice)}\n'
                f'{{location}}'
                f'Session: {session_id[:8] + "..." if session_id else "none (not counted)"}\n'
                f'\n'
                f'Total sequences classified: {stats_now.get("total_sequences", 0)}\n'
                f'Total prediction runs: {stats_now.get("total_runs", 0)}\n'
                f'Unique visitors: {stats_now.get("unique_sessions", 0)}'
            )
            record_usage_location(_client_ip(), message_template)

            resp = jsonify({
                'model': model_choice,
                'num_sequences': len(sequences),
                'predictions': predictions_df.to_dict(orient='records')
            })
            # um navegador que chegou aqui sem cookie sai com um: a rodada
            # atual nao conta como visitante, a proxima conta. curl e bot nao
            # guardam cookie, entao continuam de fora.
            if not session_id:
                resp.set_cookie('_amp_sid', str(uuid.uuid4()),
                                max_age=365 * 24 * 3600, samesite='Lax', httponly=True)
            return resp

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
