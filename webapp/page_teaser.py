"""Teaser served at /beta.

A letter soup in glass that settles into one sentence and gets out of the
way of the pointer. React drives it because the displacement is a
per-letter distance computed every frame, which no stylesheet can do; the
rest is the design language untouched.

The predictor that used to live here is still in page_beta.py, unrouted.
Pointing /beta back at it is one line in app.py.
"""

from webapp.page_shell import page

PHRASE = 'sooooooooonnnnnnnn'

BODY = """
  <main class="step-2 soup-main" id="main" tabindex="-1">
    <div class="soup" id="soup" role="img" aria-label="More is yet to come, soon"></div>

    <!-- A piada nao se explica: um residuo mascarado no meio da LL-37 e a
         tarefa com que um modelo de linguagem de proteina e treinado, e a
         posicao 7 e um R para quem quiser conferir. Quem conhece a area
         le o recado inteiro; quem nao conhece ve uma sequencia. -->
    <p class="soup-mask">
      <span class="soup-seq">LLGDFF<button type="button" class="soup-token hit"
             aria-label="Masked residue, position 7 of LL-37. Reveal it: arginine.">
        <span class="mask-face mask-face-hidden" aria-hidden="true">[MASK]</span>
        <span class="mask-face mask-face-shown" aria-hidden="true">R</span>
      </button>KSKEKIGKEFKRIVQRIKDFLRNLVPRTES</span>
      <span class="soup-aside" role="status">
        <span class="mask-face mask-line-rest" aria-hidden="true" id="asideScramble" data-text="Something that fills that in is in training."></span>
        <span class="mask-face mask-line-done" aria-hidden="true">You just did what it is being trained to do.</span>
      </span>
    </p>

    <noscript>
      <p class="soup-fallback">More is yet to come.</p>
    </noscript>
  </main>
"""

CSS = """
  /* A pagina inteira e uma frase, entao ela ocupa o que sobra entre as
     duas barras fixas em vez de comecar colada no topo. */
  .soup-main {
    min-height: calc(100vh - var(--chrome-top) - var(--chrome-bottom) - var(--space-96) - var(--space-96));
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-48);
  }

  /* o residuo mascarado fica em tinta cheia e o resto da cadeia em tinta
     apagada: o que importa e o buraco, nao a sequencia */
  .soup-mask {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-8);
    text-align: center;
  }

  .soup-seq { font-family: var(--font-mono); font-size: var(--text-13); color: var(--muted); letter-spacing: var(--tracking-wide); }

  /* O residuo mascarado e um controle, nao um enfeite: e o gesto que
     completa a piada. Largura fixa na maior das duas faces, senao a
     cadeia inteira anda para o lado quando [MASK] vira R. */
  .soup-token {
    display: inline-grid;
    place-items: center;
    position: relative;
    min-width: var(--space-48);
    padding: 0;
    border: none;
    border-bottom: var(--stroke) dotted var(--secondary);
    background: none;
    font: inherit;
    color: var(--text);
    cursor: pointer;
    vertical-align: baseline;
  }

  .mask-face {
    grid-area: 1 / 1;
    transition:
      opacity var(--duration-3) var(--ease-out-expo),
      transform var(--duration-3) var(--ease-out-expo),
      filter var(--duration-3) var(--ease-out-expo);
  }

  .mask-face-hidden, .mask-line-rest { opacity: 1; transform: scale(1); filter: blur(0); }
  .mask-face-shown, .mask-line-done { opacity: 0; transform: scale(0.25); filter: blur(var(--motion-blur-2)); }

  .soup-token:hover .mask-face-hidden,
  .soup-token:focus-visible .mask-face-hidden { opacity: 0; transform: scale(0.25); filter: blur(var(--motion-blur-2)); }
  .soup-token:hover .mask-face-shown,
  .soup-token:focus-visible .mask-face-shown { opacity: 1; transform: scale(1); filter: blur(0); }

  /* a legenda troca junto, e por isso a piada se explica sozinha para
     quem nao a pegou de primeira. Sem :has() o token ainda troca, so a
     legenda fica parada: a degradacao perde a explicacao, nao o gesto. */

  /* a legenda embaralha com o mesmo desenho do slogan da raiz: letra
     assentada em tinta cheia, letra girando em --muted */
  .soup-aside { font-family: var(--font-mono); }
  .mask-line-rest .done { color: var(--text); }
  .mask-line-rest.is-out { opacity: 0; }

  .soup-aside {
    display: inline-grid;
    font-size: var(--text-13);
    color: var(--muted);
  }

  .soup-mask:has(.soup-token:hover) .mask-line-rest,
  .soup-mask:has(.soup-token:focus-visible) .mask-line-rest { opacity: 0; transform: scale(0.25); filter: blur(var(--motion-blur-2)); }
  .soup-mask:has(.soup-token:hover) .mask-line-done,
  .soup-mask:has(.soup-token:focus-visible) .mask-line-done { opacity: 1; transform: scale(1); filter: blur(0); }

  .soup {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: var(--space-10) var(--space-24);
    max-width: var(--measure-wide);
  }

  /* uma palavra nao quebra no meio: o espaco entre palavras e a folga do
     grupo, e dentro do grupo as letras ficam juntas */
  .soup-word { display: flex; gap: var(--space-4); }

  /* o vao entre palavras e o do contentor e tem de ser maior que o vao
     entre letras, senao a frase le como uma palavra so */

  /* Os tres tokens sao registrados porque custom property nao registrada
     e texto: ela salta entre quadros e nao interpola, e o assentamento
     depois que o ponteiro sai leria como teletransporte. */
  @property --soup-x { syntax: '<length>'; inherits: false; initial-value: 0px; }
  @property --soup-y { syntax: '<length>'; inherits: false; initial-value: 0px; }
  @property --soup-rot { syntax: '<angle>'; inherits: false; initial-value: 0deg; }

  .soup-letter {
    display: grid;
    place-items: center;
    width: var(--soup-size);
    height: var(--soup-size);
    border-radius: var(--radius-field);
    font-family: var(--font-display);
    font-stretch: var(--font-display-stretch);
    font-weight: var(--weight-light);
    font-size: var(--text-32);
    line-height: var(--leading-none);
    color: var(--text);
    user-select: none;
    /* UMA declaracao de transform, composta dos tres tokens. Duas regras
       escrevendo transform no mesmo elemento e o defeito que a camada
       liquida ja pagou uma vez. Toda reserva de var() esta presente:
       sem @property o calc fica invalido e a declaracao inteira cai
       para none, levando junto o repouso. */
    transform:
      translate(var(--soup-x, 0px), var(--soup-y, 0px))
      rotate(var(--soup-rot, 0deg));
  }

  /* Duas classes, e nao uma, porque `.glass:not(.card-glass)` pesa duas e
     declara transition propria: com um seletor de uma classe a transicao
     do ladrilho some inteira e ele passa a saltar em vez de assentar.
     Medido: transitionProperty voltava com as quatro do vidro e nenhuma
     das minhas. A lista repete as do vidro porque substituir a
     declaracao substitui a lista toda. */
  .soup .soup-letter {
    transition:
      transform var(--duration-5) var(--ease-out-soft),
      opacity var(--duration-4) var(--ease-out-soft),
      background-image var(--duration-5) var(--ease-out-soft),
      border-color var(--duration-5) var(--ease-out-soft),
      box-shadow var(--duration-5) var(--ease-out-soft),
      color var(--duration-4) var(--ease-out-soft);
  }

  .soup-size-sm { --soup-size: var(--space-40); }

  /* o espaco entre palavras nao e ladrilho: nao tem vidro, nao recebe
     ponteiro e nao entra na conta de deslocamento */
  .soup-gap { width: var(--space-16); }

  .soup-fallback {
    font-family: var(--font-display);
    font-stretch: var(--font-display-stretch);
    font-weight: var(--weight-light);
    font-size: var(--text-32);
    color: var(--text);
  }

  @media (max-width: 768px) {
    .soup-letter { font-size: var(--text-20); }
    .soup-size-sm { --soup-size: var(--space-32); }
  }

  /* Sem ponteiro fino nao ha de onde fugir, e sob movimento reduzido a
     frase fica e o movimento sai. Nos dois casos o repouso e o desenho
     final, nao um estado degradado. */
  @media (pointer: coarse), (prefers-reduced-motion: reduce) {
    .soup-letter { transition: none; }
  }
"""

JS = """
/* ---------- a legenda se resolve como o slogan da raiz ----------
   Mesmo alfabeto e mesmo ciclo da frase que abre a pagina principal:
   uma coisa so, escrita em dois lugares. Aqui ela para enquanto o
   ponteiro esta sobre o residuo mascarado, porque nesse momento a
   legenda ja foi trocada pela outra face e embaralhar o que esta
   escondido e trabalho jogado fora. */

(function () {
  var AMINO = 'ACDEFGHIKLMNPQRSTVWY';
  var TICK = 45;
  var PER_CHAR = 2;
  var HOLD = 3200;
  var GAP = 500;

  var host = document.getElementById('asideScramble');
  var token = document.querySelector('.soup-token');
  if (!host) return;
  var text = host.dataset.text || '';
  var still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (still) { host.textContent = text; return; }

  var timer = null;
  var parado = false;

  function rand() { return AMINO[Math.floor(Math.random() * AMINO.length)]; }

  function frame(resolvidas) {
    var out = '';
    for (var i = 0; i < text.length; i++) {
      var ch = text[i];
      if (ch === ' ' || ch === '.') { out += ch; continue; }
      out += i < resolvidas ? '<span class="done">' + ch + '</span>' : rand();
    }
    host.innerHTML = out;
  }

  function run() {
    if (parado) return;
    var passo = 0;
    host.classList.remove('is-out');
    timer = window.setInterval(function () {
      var resolvidas = Math.floor(passo / PER_CHAR);
      frame(resolvidas);
      passo++;
      if (resolvidas > text.length) {
        window.clearInterval(timer);
        frame(text.length);
        timer = window.setTimeout(function () {
          host.classList.add('is-out');
          timer = window.setTimeout(run, GAP);
        }, HOLD);
      }
    }, TICK);
  }

  function stop(fixa) {
    window.clearInterval(timer);
    window.clearTimeout(timer);
    timer = null;
    if (fixa) { host.classList.remove('is-out'); host.textContent = text; }
  }

  if (token) {
    token.addEventListener('pointerenter', function () { parado = true; stop(true); });
    token.addEventListener('focus', function () { parado = true; stop(true); });
    token.addEventListener('pointerleave', function () { parado = false; run(); });
    token.addEventListener('blur', function () { parado = false; run(); });
  }

  document.addEventListener('visibilitychange', function () {
    if (document.hidden) stop(true);
    else if (!timer && !parado) run();
  });

  run();
})();

/* ---------- sopa de letrinhas ----------
   React so existe aqui pelo que a folha de estilo nao alcanca: a
   distancia de cada ladrilho ate o ponteiro, recalculada por quadro. O
   componente escreve tres custom properties e nada mais; quem decide
   aparencia continua sendo o CSS, como na camada de luz.

   Os numeros abaixo sao de fisica, nao de desenho: alcance em pixels,
   forca em pixels, giro em graus. Nao viram token porque nenhum deles e
   lido por uma folha de estilo. */

(function () {
  var PHRASE = 'More is yet to come soon';
  var REACH = 160;
  var PUSH = 34;
  var TILT = 14;

  var h = React.createElement;
  var fine = window.matchMedia('(pointer: fine)').matches;
  var still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function Letter(props) {
    var ref = React.useRef(null);
    React.useEffect(function () { props.register(props.index, ref.current); }, []);
    return h('span', {
      ref: ref,
      className: 'soup-letter glass glass-frost soup-size-sm',
      'aria-hidden': 'true'
    }, props.char);
  }

  function Soup() {
    var nodes = React.useRef([]);
    var boxes = React.useRef([]);
    var frame = React.useRef(0);

    var register = React.useCallback(function (i, el) { nodes.current[i] = el; }, []);

    var measure = React.useCallback(function () {
      boxes.current = nodes.current.map(function (el) {
        if (!el) return null;
        var b = el.getBoundingClientRect();
        return { x: b.left + b.width / 2, y: b.top + b.height / 2 };
      });
    }, []);

    var settle = React.useCallback(function () {
      nodes.current.forEach(function (el) {
        if (!el) return;
        el.style.setProperty('--soup-x', '0px');
        el.style.setProperty('--soup-y', '0px');
        el.style.setProperty('--soup-rot', '0deg');
      });
    }, []);

    React.useEffect(function () {
      measure();
      if (!fine || still) return;

      var pointer = null;

      function paint() {
        frame.current = 0;
        if (!pointer) return;
        for (var i = 0; i < nodes.current.length; i++) {
          var el = nodes.current[i];
          var box = boxes.current[i];
          if (!el || !box) continue;
          var dx = box.x - pointer.x;
          var dy = box.y - pointer.y;
          var d = Math.sqrt(dx * dx + dy * dy);
          if (d > REACH || d === 0) {
            /* escrita repetida e descartada: fora do alcance o ladrilho
               ja esta em repouso e nao ha o que escrever */
            if (el.dataset.moved === '1') {
              el.style.setProperty('--soup-x', '0px');
              el.style.setProperty('--soup-y', '0px');
              el.style.setProperty('--soup-rot', '0deg');
              el.dataset.moved = '0';
            }
            continue;
          }
          var near = 1 - d / REACH;
          el.style.setProperty('--soup-x', (dx / d * PUSH * near).toFixed(2) + 'px');
          el.style.setProperty('--soup-y', (dy / d * PUSH * near).toFixed(2) + 'px');
          el.style.setProperty('--soup-rot', ((dx > 0 ? 1 : -1) * TILT * near).toFixed(2) + 'deg');
          el.dataset.moved = '1';
        }
      }

      function onMove(ev) {
        pointer = { x: ev.clientX, y: ev.clientY };
        if (!frame.current) frame.current = requestAnimationFrame(paint);
      }

      function onLeave() { pointer = null; settle(); }

      function onResize() { measure(); settle(); }

      window.addEventListener('pointermove', onMove, { passive: true });
      window.addEventListener('pointerleave', onLeave);
      window.addEventListener('blur', onLeave);
      window.addEventListener('resize', onResize);
      window.addEventListener('scroll', measure, { passive: true });

      return function () {
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerleave', onLeave);
        window.removeEventListener('blur', onLeave);
        window.removeEventListener('resize', onResize);
        window.removeEventListener('scroll', measure);
        if (frame.current) cancelAnimationFrame(frame.current);
      };
    }, [measure, settle]);

    var index = 0;
    var words = PHRASE.split(' ').map(function (word, w) {
      var letters = word.split('').map(function (ch) {
        var i = index++;
        return h(Letter, { key: i, index: i, char: ch, register: register });
      });
      return h('span', { className: 'soup-word', key: 'w' + w }, letters);
    });

    return h(React.Fragment, null, words);
  }

  var host = document.getElementById('soup');
  if (host && window.React && window.ReactDOM) {
    ReactDOM.createRoot(host).render(h(Soup));
  } else if (host) {
    /* React nao carregou: a frase entra como texto e a pagina continua
       dizendo o que veio dizer */
    host.textContent = 'Soon.';
    host.className = 'soup-fallback';
  }
})();
"""

HEAD_EXTRA = (
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.3.1/umd/'
    'react.production.min.js" crossorigin></script>\n'
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.3.1/umd/'
    'react-dom.production.min.js" crossorigin></script>\n'
)

COPY = {
    'title': {'en': 'AMPidentifier BETA | More is yet to come',
              'pt': 'AMPidentifier BETA | Ainda vem mais por aí'},
    'description': {
        'en': ('The AMPidentifier beta. A new batch of trained models and a '
               'prediction mode built on a protein language model are in '
               'testing. The stable predictor runs at ampidentifier.com.'),
        'pt': ('A beta do AMPidentifier. Um novo conjunto de modelos treinados e '
               'um modo de predição sobre modelo de linguagem de proteína estão '
               'em teste. O preditor estável roda em ampidentifier.com.')},
    # A frase da sopa nao se traduz: ela e a marca da pagina e as quinze
    # letras sao o desenho. O que se traduz e o que a explica.
    'aside': {'en': 'Something that fills that in is in training.',
              'pt': 'Algo que preenche isso está em treinamento.'},
    'done': {'en': 'You just did what it is being trained to do.',
             'pt': 'Você acabou de fazer o que ele está aprendendo a fazer.'},
    'mask_label': {'en': 'Masked residue, position 7 of LL-37. Reveal it: arginine.',
                   'pt': 'Resíduo mascarado, posição 7 da LL-37. Revele: arginina.'},
    'fallback': {'en': 'Soon.', 'pt': 'Ainda vem mais por aí.'},
}


def build(lang):
    c = {k: v[lang] for k, v in COPY.items()}
    body = (BODY
            .replace('Something that fills that in is in training.', c['aside'])
            .replace('You just did what it is being trained to do.', c['done'])
            .replace('Masked residue, position 7 of LL-37. Reveal it: arginine.', c['mask_label'])
            .replace('>More is yet to come.<', '>' + c['fallback'] + '<'))
    return page(title=c['title'], description=c['description'], path='/beta',
                body=body, css=CSS, js=JS, schema=HEAD_EXTRA, lang=lang)


PAGE = build('en')
PAGE_PT = build('pt')
