/* Pure Design 1.6.1 -- copia de /Volumes/promethion/pure-design-language/web/light.js
 pure / light.js
   O unico javascript da linguagem, e ele existe por uma razao que o CSS
   nao contorna: a posicao do ponteiro nao e legivel em folha de estilo.

   O que este arquivo faz, inteiro:

     mede uma caixa e escreve tres numeros nela.

     --light-x     posicao horizontal do ponteiro dentro da caixa, 0 a 1
     --light-y     posicao vertical, 0 a 1
     --light-near  proximidade, 1 sob o ponteiro e 0 a --light-reach de
                   distancia da borda

   O que ele NAO faz, e a lista importa tanto quanto a de cima: ele nao
   conhece realce, aro, inclinacao, ima nem inchaco. Nenhum nome de
   efeito aparece neste arquivo. Toda a aparencia mora em light.css, e e
   por isso que um efeito novo na camada de luz nao toca em javascript.

   USO

     <script src="pure/light.js" defer></script>

     <button class="pill glass lit lit-swell" data-lit>...</button>

   `data-lit` marca quem e medido. Elemento sem o atributo nao custa
   nada: a varredura nem o enxerga.

   Depois de inserir marcacao nova, nada a fazer: um MutationObserver
   revarre sozinho. `window.pureLight.scan()` existe para o caso de uma
   insercao fora do DOM observado, e `stop()` desliga tudo.

   CUSTO, medido e limitado de proposito:

   - um unico listener de ponteiro na janela, nao um por elemento
   - as escritas acontecem num rAF, nunca no evento: varios movimentos
     no mesmo quadro viram uma escrita so
   - getBoundingClientRect e o unico ponto de layout forcado, entao ele
     roda em lote e so quando a geometria pode ter mudado (rolagem,
     redimensionamento, mutacao). Medir por elemento dentro do laco de
     ponteiro e o jeito classico de o efeito custar 8ms por quadro
   - elemento fora do alcance sai do laco antes de qualquer escrita, e
     escrita repetida do mesmo valor e descartada: parado, o custo por
     quadro e zero

   DEGRADACAO: em `pointer: coarse` nada se registra. Nao ha pairar num
   dedo, e a mesma consulta zera a camada em light.css, entao os dois
   lados concordam sem depender um do outro. */

(function () {
  "use strict";

  var doc = document;
  var root = doc.documentElement;

  /* a mesma consulta que light.css usa. Sem ponteiro fino a camada
     inteira nao existe, e o script nao se registra em nada. */
  if (window.matchMedia && window.matchMedia("(pointer: coarse)").matches) return;

  var tracked = [];       /* { el, rect, reach, x, y, near } */
  var cursors = [];
  var px = 0, py = 0;     /* ponteiro, em coordenadas de viewport */
  var seen = false;       /* o ponteiro ja apareceu nesta pagina */
  var dirty = true;       /* a geometria pode ter mudado */
  var frame = 0;
  var rescan = 0;
  var running = false;

  /* --light-reach e um token: ele pode ser trocado por elemento, por
     secao ou por modo. Ler no scan e nao no laco de ponteiro, porque
     getComputedStyle dentro do laco custa o mesmo que medir a caixa. */
  function reachOf(el) {
    var raw = getComputedStyle(el).getPropertyValue("--light-reach").trim();
    var n = parseFloat(raw);
    return isFinite(n) && n > 0 ? n : 240;
  }

  function scan() {
    tracked = [];
    var nodes = doc.querySelectorAll("[data-lit]");
    for (var i = 0; i < nodes.length; i++) {
      tracked.push({ el: nodes[i], rect: null, reach: reachOf(nodes[i]), x: -1, y: -1, near: -1 });
    }
    cursors = [].slice.call(doc.querySelectorAll(".lit-cursor"));
    dirty = true;
    request();
  }

  /* uma pagina viva muta muitas vezes por quadro, e scan() paga
     querySelectorAll mais um getComputedStyle por elemento. Sem esta
     fila a camada custa mais em pagina parada com lista virtual do que
     custa seguindo o cursor. */
  function queueScan() {
    if (rescan) return;
    rescan = requestAnimationFrame(function () {
      rescan = 0;
      scan();
    });
  }

  function measure() {
    for (var i = 0; i < tracked.length; i++) tracked[i].rect = tracked[i].el.getBoundingClientRect();
    dirty = false;
  }

  /* escrita descartada quando o valor nao mudou de verdade. Tres casas
     e mais resolucao do que qualquer tela usa e menos do que basta para
     um arredondamento de subpixel reescrever o token a cada quadro. */
  function put(entry, prop, key, value) {
    var v = Math.round(value * 1000) / 1000;
    if (entry[key] === v) return;
    entry[key] = v;
    entry.el.style.setProperty(prop, String(v));
  }

  function tick() {
    frame = 0;
    if (!seen) return;
    if (dirty) measure();

    for (var i = 0; i < tracked.length; i++) {
      var t = tracked[i];
      var r = t.rect;
      if (!r || (r.width === 0 && r.height === 0)) continue;

      /* distancia do ponteiro ate a caixa, zero quando ele esta dentro.
         Medir ate a BORDA e nao ate o centro e o que faz um cartao largo
         e um botao pequeno acenderem na mesma distancia de aproximacao. */
      var dx = px < r.left ? r.left - px : px > r.right ? px - r.right : 0;
      var dy = py < r.top ? r.top - py : py > r.bottom ? py - r.bottom : 0;
      var dist = dx === 0 && dy === 0 ? 0 : Math.sqrt(dx * dx + dy * dy);

      if (dist >= t.reach) {
        /* fora do alcance: zera a proximidade uma vez e nao escreve
           mais nada. A posicao fica onde estava de proposito, senao o
           realce salta para o centro ao apagar. */
        put(t, "--light-near", "near", 0);
        continue;
      }

      put(t, "--light-x", "x", (px - r.left) / (r.width || 1));
      put(t, "--light-y", "y", (py - r.top) / (r.height || 1));
      put(t, "--light-near", "near", 1 - dist / t.reach);
    }

    for (var j = 0; j < cursors.length; j++) {
      var c = cursors[j].style;
      c.setProperty("--light-shift-x", px + "px");
      c.setProperty("--light-shift-y", py + "px");
      if (c.getPropertyValue("--light-near") !== "1") c.setProperty("--light-near", "1");
    }
  }

  function request() {
    if (!frame) frame = requestAnimationFrame(tick);
  }

  function onMove(e) {
    px = e.clientX;
    py = e.clientY;
    seen = true;
    request();
  }

  /* o ponteiro saindo da janela apaga a camada inteira. Sem isto o
     realce fica congelado na ultima posicao quando o cursor vai para
     outra aba, que le como um defeito de pintura. */
  function onLeave() {
    seen = false;
    for (var i = 0; i < tracked.length; i++) put(tracked[i], "--light-near", "near", 0);
    for (var j = 0; j < cursors.length; j++) cursors[j].style.setProperty("--light-near", "0");
  }

  function onGeometry() {
    dirty = true;
    request();
  }

  var observer = null;

  function start() {
    if (running) return;
    running = true;
    window.addEventListener("pointermove", onMove, { passive: true });
    doc.addEventListener("pointerleave", onLeave);
    window.addEventListener("blur", onLeave);
    window.addEventListener("scroll", onGeometry, { passive: true, capture: true });
    window.addEventListener("resize", onGeometry, { passive: true });
    if (window.MutationObserver) {
      observer = new MutationObserver(queueScan);
      observer.observe(doc.body, { childList: true, subtree: true });
    }
    scan();
  }

  function stop() {
    running = false;
    window.removeEventListener("pointermove", onMove);
    doc.removeEventListener("pointerleave", onLeave);
    window.removeEventListener("blur", onLeave);
    window.removeEventListener("scroll", onGeometry, true);
    window.removeEventListener("resize", onGeometry);
    if (observer) { observer.disconnect(); observer = null; }
    if (rescan) { cancelAnimationFrame(rescan); rescan = 0; }
    if (frame) { cancelAnimationFrame(frame); frame = 0; }
    onLeave();
    tracked = [];
    cursors = [];
  }

  window.pureLight = { scan: scan, start: start, stop: stop };

  if (doc.readyState === "loading") doc.addEventListener("DOMContentLoaded", start);
  else start();
})();
