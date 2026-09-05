"""Teaser served at /beta.

A letter soup in glass that settles into one sentence and gets out of the
way of the pointer. React drives it because the displacement is a
per-letter distance computed every frame, which no stylesheet can do; the
rest is the design language untouched.

The predictor that used to live here is still in page_beta.py, unrouted.
Pointing /beta back at it is one line in app.py.
"""

from webapp.page_shell import page

PHRASE = 'more is yet to come'

BODY = """
  <main class="step-2 soup-main" id="main" tabindex="-1">
    <div class="soup" id="soup" role="img" aria-label="More is yet to come"></div>
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
    align-items: center;
    justify-content: center;
  }

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
/* ---------- sopa de letrinhas ----------
   React so existe aqui pelo que a folha de estilo nao alcanca: a
   distancia de cada ladrilho ate o ponteiro, recalculada por quadro. O
   componente escreve tres custom properties e nada mais; quem decide
   aparencia continua sendo o CSS, como na camada de luz.

   Os numeros abaixo sao de fisica, nao de desenho: alcance em pixels,
   forca em pixels, giro em graus. Nao viram token porque nenhum deles e
   lido por uma folha de estilo. */

(function () {
  var PHRASE = 'more is yet to come';
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
    host.textContent = 'More is yet to come.';
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

PAGE = page(
    title='AMPidentifier BETA | More is yet to come',
    description=('The AMPidentifier beta. A new batch of trained models and a '
                 'prediction mode built on a protein language model are in '
                 'testing. The stable predictor runs at ampidentifier.com.'),
    path='/beta',
    body=BODY,
    css=CSS,
    js=JS,
    schema=HEAD_EXTRA,
)
