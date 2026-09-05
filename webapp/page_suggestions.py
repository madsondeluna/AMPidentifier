"""Suggestions page, served at /suggestions.

The form posts to /send_suggestion, which mails the developer through the
same Resend path already used by /send_recommendation. The GitHub issue
modal stays where it is: it is the route for a bug with a reproduction,
this one is the route for everything else.
"""

from webapp.page_shell import page

BODY = """
  <header class="step-1">
    <h1 class="page-title">Suggestions</h1>
  </header>

  <main class="step-2" id="main" tabindex="-1">

    <div class="card-glass prose-block prose-justify">
      <p>Feature requests, missing models, wrong results, anything about the
      interface. The message goes straight to the developer. A reply address is
      optional and is used only to answer this message.</p>
    </div>

    <form class="surface form-block step-2" id="suggestionForm" novalidate>
      <label class="field">
        <span class="field-label" for="sugTopic">Topic</span>
        <span class="select-shell">
          <select class="select" id="sugTopic">
            <option value="feature">Feature request</option>
            <option value="model">Model or prediction quality</option>
            <option value="interface">Interface</option>
            <option value="data">Data or export</option>
            <option value="other">Other</option>
          </select>
        </span>
      </label>

      <label class="field">
        <span class="field-label" for="sugMsg">Message</span>
        <textarea class="textarea" id="sugMsg" rows="6"
                  placeholder="What should change, and why"></textarea>
      </label>

      <label class="field">
        <span class="field-label" for="sugEmail">Your email, to get a reply</span>
        <input class="input" type="email" id="sugEmail" autocomplete="email"
               spellcheck="false" placeholder="you@example.com">
      </label>

      <div class="form-actions">
        <button class="pill glass-accent" type="submit" id="sugSend">Send</button>
        <span class="form-status" id="sugStatus" aria-live="polite"></span>
      </div>
    </form>

    <div class="metrics-band step-2">
      <div class="metrics-label">Other routes</div>
      <div class="card-glass prose-block prose-justify">
        <p class="install"><span class="install-lead">Bug with a reproduction:</span> <a href="https://github.com/madsondeluna/AMPidentifier/issues" target="_blank" rel="noopener">open an issue on GitHub</a></p>
        <p class="install"><span class="install-lead">Direct email:</span> <a href="mailto:madsondeluna@gmail.com">madsondeluna@gmail.com</a></p>
      </div>
    </div>

  </main>
"""

CSS = """
  .page-title { font-size: var(--text-32); }
  .prose-block > p + p { margin-top: var(--space-12); }
  .prose-block { padding: var(--space-16) var(--space-24); }
  .form-block { display: flex; flex-direction: column; gap: var(--space-16); padding: var(--space-24); }
  .form-actions { display: flex; align-items: center; gap: var(--space-16); flex-wrap: wrap; }
  .form-status { font-size: var(--text-13); color: var(--muted); }
  .form-status.ok { color: var(--status-good); }
"""

JS = """
/* O botao mantem o rotulo e ganha o anel enquanto envia: trocar o texto
   por "Enviando" muda a largura do controle e a linha toda se mexe. */
document.getElementById('suggestionForm').addEventListener('submit', async function (ev) {
  ev.preventDefault();
  const msg = document.getElementById('sugMsg');
  const email = document.getElementById('sugEmail');
  const status = document.getElementById('sugStatus');
  const btn = document.getElementById('sugSend');

  status.classList.remove('ok');
  if (!msg.value.trim()) {
    status.innerHTML = '<span class="err">Write a message before sending.</span>';
    msg.focus();
    return;
  }
  if (email.value.trim() && !/^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(email.value.trim())) {
    status.innerHTML = '<span class="err">That email address is not valid.</span>';
    email.focus();
    return;
  }

  btn.disabled = true;
  status.textContent = 'Sending';
  try {
    const fd = new FormData();
    fd.append('topic', document.getElementById('sugTopic').value);
    fd.append('message', msg.value.trim());
    fd.append('reply_to', email.value.trim());
    const res = await fetch('/send_suggestion', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.ok) {
      status.classList.add('ok');
      status.textContent = 'Sent. Thank you.';
      msg.value = '';
    } else {
      status.innerHTML = '<span class="err">' + (data.error || 'Failed to send.') + '</span>';
    }
  } catch (e) {
    status.innerHTML = '<span class="err">' + e.message + '</span>';
  } finally {
    btn.disabled = false;
  }
});
"""

# ---------------------------------------------------------------------------
# Versao em portugues. A traducao e por substituicao sobre o mesmo corpo,
# com asserto em cada par: se o texto de origem mudar e o par deixar de
# casar, o modulo falha ao importar em vez de servir uma pagina meio
# traduzida.
# ---------------------------------------------------------------------------

BODY_PAIRS = [
    ('Suggestions</h1>', 'Sugest&otilde;es</h1>'),
    ('>Topic</span>', '>Assunto</span>'),
    ('>Feature request</option>', '>Pedido de funcionalidade</option>'),
    ('>Model or prediction quality</option>', '>Modelo ou qualidade da predi&ccedil;&atilde;o</option>'),
    ('>Data or export</option>', '>Dados ou exporta&ccedil;&atilde;o</option>'),
    ('>Other</option>', '>Outro</option>'),
    ('>Message</span>', '>Mensagem</span>'),
    ('placeholder="What should change, and why"',
     'placeholder="O que deveria mudar, e por qu&ecirc;"'),
    ('>Your email, to get a reply</span>', '>Seu email, para receber resposta</span>'),
    ('>Send</button>', '>Enviar</button>'),
    ('>Other routes</div>', '>Outros caminhos</div>'),
    ('>Bug with a reproduction:</span>', '>Defeito com passos para reproduzir:</span>'),
    ('>open an issue on GitHub</a>', '>abra uma issue no GitHub</a>'),
    ('>Direct email:</span>', '>Email direto:</span>'),
]

BODY_LONG = (
    'Feature requests, missing models, wrong results, anything about the',
    'Pedidos de funcionalidade, modelos que faltam, resultados errados, qualquer',
)

JS_PAIRS = [
    ('Write a message before sending.', 'Escreva uma mensagem antes de enviar.'),
    ('That email address is not valid.', 'Esse endere\u00e7o de email n\u00e3o \u00e9 v\u00e1lido.'),
    ("'Sending'", "'Enviando'"),
    ("'Sent. Thank you.'", "'Enviado. Obrigado.'"),
    ("'Failed to send.'", "'Falha ao enviar.'"),
]

BODY_PT = BODY
for _a, _b in BODY_PAIRS:
    assert _a in BODY_PT, _a
    BODY_PT = BODY_PT.replace(_a, _b, 1)

_INTRO_EN = ('Feature requests, missing models, wrong results, anything about the\n'
             '      interface. The message goes straight to the developer. A reply address is\n'
             '      optional and is used only to answer this message.')
_INTRO_PT = ('Pedidos de funcionalidade, modelos que faltam, resultados errados, qualquer\n'
             '      coisa sobre a interface. A mensagem vai direto para o desenvolvedor. O\n'
             '      endere&ccedil;o de resposta &eacute; opcional e serve s&oacute; para responder a esta mensagem.')
assert _INTRO_EN in BODY_PT
BODY_PT = BODY_PT.replace(_INTRO_EN, _INTRO_PT, 1)

JS_PT = JS
for _a, _b in JS_PAIRS:
    assert _a in JS_PT, _a
    JS_PT = JS_PT.replace(_a, _b, 1)


PAGE = page(
    title='Suggestions | AMPidentifier',
    description=('Send a feature request, a note on prediction quality or an '
                 'interface problem straight to the developer of AMPidentifier.'),
    path='/suggestions',
    body=BODY,
    css=CSS,
    js=JS,
)

PAGE_PT = page(
    title='Sugest\u00f5es | AMPidentifier',
    description=('Envie um pedido de funcionalidade, uma observa\u00e7\u00e3o sobre a '
                 'qualidade da predi\u00e7\u00e3o ou um problema de interface direto para '
                 'o desenvolvedor do AMPidentifier.'),
    path='/suggestions',
    body=BODY_PT,
    css=CSS,
    js=JS_PT,
    lang='pt',
)
