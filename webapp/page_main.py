"""Main front end: the Pure Design page, served at /.

This is the page the site serves at the root. The previous markup is
still in app.py as PAGE, unrouted, so the old layout can be restored by
pointing / back at it.

O head, a folha, a barra de cima, a barra de baixo e o modal moram em
page_shell.py e sao os mesmos de /about e /suggestions. Aqui fica so o
miolo desta pagina e o javascript dela.
"""

from webapp.page_shell import page

BODY = """
<header>
    <div>
      <!-- a marca e a imagem; o titulo da pagina existe para leitor de tela
           e para a hierarquia do documento, que nao tinha nenhum h1 -->
      <h1 class="sr-only">AMPidentifier, antimicrobial peptide prediction</h1>
    </div>

    <div class="hero">
      <svg class="hero-fan" viewBox="0 0 679 210" fill="none" role="img" aria-label="AMPidentifier" xmlns="http://www.w3.org/2000/svg"> <g id="04 Export fan"> <g id="ampidentifier-compact-fan"> <g id="Group"> <circle id="Model 1" cx="102" cy="20.134" r="13.134" fill="#0E6E66"/> <rect id="Vote 1" x="98.7165" y="60.5464" width="23.2371" height="6.56701" rx="3.28351" transform="rotate(-90 98.7165 60.5464)" fill="#0E6E66"/> <circle id="Model 2" cx="41.9907" cy="44.9907" r="13.134" fill="#0E6E66"/> <rect id="Vote 2" x="68.2448" y="75.8884" width="23.2371" height="6.56701" rx="3.28351" transform="rotate(-135 68.2448 75.8884)" fill="#0E6E66"/> <circle id="Model 3" cx="17.134" cy="105" r="13.134" fill="#0E6E66"/> <rect id="Vote 3" x="57.5464" y="108.283" width="23.2371" height="6.56701" rx="3.28351" transform="rotate(-180 57.5464 108.283)" fill="#0E6E66"/> <circle id="Model 4" cx="41.9907" cy="165.009" r="13.134" fill="#0E6E66"/> <rect id="Vote 4" x="72.8883" y="138.755" width="23.2371" height="6.56701" rx="3.28351" transform="rotate(135 72.8883 138.755)" fill="#0E6E66"/> <circle id="Model 5" cx="102" cy="189.866" r="13.134" fill="#0E6E66"/> <rect id="Vote 5" x="105.284" y="149.454" width="23.2371" height="6.56701" rx="3.28351" transform="rotate(90 105.284 149.454)" fill="#0E6E66"/> <circle id="Call" cx="102" cy="105" r="26.268" fill="#0E6E66"/> </g> <path id="AMPidentifier" d="M150.728 136L175.292 72.888H190.38L215.036 136H202.064L197.004 122.568H168.116L163.056 136H150.728ZM171.888 112.448H193.14L186.792 95.336C186.547 94.7227 186.24 93.956 185.872 93.036C185.565 92.0547 185.228 91.012 184.86 89.908C184.492 88.7427 184.124 87.5773 183.756 86.412C183.388 85.2467 183.051 84.204 182.744 83.284H182.284C181.916 84.5107 181.456 85.9213 180.904 87.516C180.413 89.1107 179.923 90.6133 179.432 92.024C178.941 93.4347 178.543 94.5387 178.236 95.336L171.888 112.448ZM220.647 136V72.888H239.507L249.443 108.032C249.811 109.259 250.179 110.639 250.547 112.172C250.915 113.705 251.252 115.177 251.559 116.588C251.927 117.999 252.203 119.225 252.387 120.268H253.123C253.245 119.348 253.429 118.213 253.675 116.864C253.92 115.515 254.196 114.073 254.503 112.54C254.871 110.945 255.269 109.412 255.699 107.94L265.635 72.888H284.311V136H272.351V104.076C272.351 101.255 272.381 98.4333 272.443 95.612C272.504 92.7907 272.565 90.368 272.627 88.344C272.749 86.32 272.811 85.124 272.811 84.756H272.075C271.952 85.308 271.645 86.504 271.155 88.344C270.664 90.184 270.143 92.1773 269.591 94.324C269.039 96.4707 268.548 98.3107 268.119 99.844L257.723 136H246.683L236.287 99.936C235.919 98.648 235.489 97.1147 234.999 95.336C234.569 93.496 234.109 91.6253 233.619 89.724C233.189 87.8227 232.791 86.1667 232.423 84.756H231.687C231.748 86.412 231.809 88.4053 231.871 90.736C231.932 93.0053 231.993 95.336 232.055 97.728C232.116 100.059 232.147 102.175 232.147 104.076V136H220.647ZM296.432 136V72.888H328.08C332.741 72.888 336.574 73.716 339.58 75.372C342.646 77.028 344.946 79.3587 346.48 82.364C348.013 85.3693 348.78 88.9573 348.78 93.128C348.78 97.2373 347.982 100.825 346.388 103.892C344.793 106.959 342.401 109.351 339.212 111.068C336.084 112.724 332.158 113.552 327.436 113.552H308.392V136H296.432ZM308.392 103.34H327.068C330.134 103.34 332.496 102.451 334.152 100.672C335.869 98.832 336.728 96.3173 336.728 93.128C336.728 90.9813 336.36 89.172 335.624 87.7C334.888 86.228 333.814 85.0933 332.404 84.296C330.993 83.4987 329.214 83.1 327.068 83.1H308.392V103.34ZM355.672 77.672V69.484H362.296V77.672H355.672ZM355.672 136V87.608H362.296V136H355.672ZM389.794 137.104C385.808 137.104 382.373 136.215 379.49 134.436C376.608 132.657 374.4 129.897 372.866 126.156C371.394 122.353 370.658 117.416 370.658 111.344C370.658 105.64 371.425 100.979 372.958 97.36C374.492 93.68 376.669 90.9507 379.49 89.172C382.373 87.3933 385.746 86.504 389.61 86.504C391.696 86.504 393.658 86.7493 395.498 87.24C397.338 87.7307 398.994 88.528 400.466 89.632C401.938 90.6747 403.226 92.0853 404.33 93.864H404.882V69.484H411.506V136H406.354L405.618 128.916H404.974C403.257 131.676 401.049 133.731 398.35 135.08C395.713 136.429 392.861 137.104 389.794 137.104ZM390.898 131.308C394.21 131.308 396.878 130.572 398.902 129.1C400.988 127.628 402.49 125.512 403.41 122.752C404.392 119.992 404.882 116.619 404.882 112.632V110.976C404.882 107.173 404.484 104.045 403.686 101.592C402.95 99.1387 401.908 97.2373 400.558 95.888C399.27 94.5387 397.829 93.6187 396.234 93.128C394.64 92.576 393.014 92.3 391.358 92.3C388.292 92.3 385.716 92.944 383.63 94.232C381.606 95.4587 380.073 97.4213 379.03 100.12C378.049 102.819 377.558 106.345 377.558 110.7V113C377.558 117.539 378.11 121.157 379.214 123.856C380.318 126.493 381.852 128.395 383.814 129.56C385.838 130.725 388.2 131.308 390.898 131.308ZM441.286 137.104C436.686 137.104 432.791 136.215 429.602 134.436C426.412 132.596 423.99 129.805 422.334 126.064C420.678 122.323 419.85 117.569 419.85 111.804C419.85 105.977 420.678 101.224 422.334 97.544C423.99 93.8027 426.443 91.0427 429.694 89.264C432.944 87.424 436.992 86.504 441.838 86.504C446.315 86.504 450.056 87.3933 453.062 89.172C456.128 90.9507 458.428 93.5267 459.962 96.9C461.556 100.273 462.354 104.413 462.354 109.32V113.552H426.75C426.75 117.784 427.332 121.249 428.498 123.948C429.663 126.647 431.319 128.609 433.466 129.836C435.674 131.001 438.372 131.584 441.562 131.584C443.892 131.584 445.916 131.277 447.634 130.664C449.412 130.051 450.854 129.223 451.958 128.18C453.123 127.076 454.012 125.788 454.626 124.316C455.239 122.844 455.546 121.311 455.546 119.716H461.986C461.986 122.108 461.526 124.377 460.606 126.524C459.747 128.609 458.459 130.449 456.742 132.044C455.086 133.639 452.97 134.896 450.394 135.816C447.818 136.675 444.782 137.104 441.286 137.104ZM426.75 108.308H455.454C455.454 105.241 455.086 102.696 454.35 100.672C453.675 98.5867 452.724 96.9307 451.498 95.704C450.271 94.416 448.799 93.496 447.082 92.944C445.426 92.392 443.555 92.116 441.47 92.116C438.464 92.116 435.888 92.6987 433.742 93.864C431.595 94.968 429.939 96.716 428.774 99.108C427.67 101.5 426.995 104.567 426.75 108.308ZM470.543 136V87.608H475.787L476.431 95.244H477.075C478.424 93.036 479.958 91.3187 481.675 90.092C483.392 88.804 485.232 87.884 487.195 87.332C489.219 86.78 491.366 86.504 493.635 86.504C496.702 86.504 499.37 87.0253 501.639 88.068C503.97 89.0493 505.779 90.6747 507.067 92.944C508.355 95.2133 508.999 98.3107 508.999 102.236V136H502.375V102.88C502.375 100.733 502.099 98.9853 501.547 97.636C500.995 96.2867 500.228 95.244 499.247 94.508C498.266 93.7107 497.1 93.1587 495.751 92.852C494.463 92.5453 493.022 92.392 491.427 92.392C488.974 92.392 486.643 93.0053 484.435 94.232C482.288 95.3973 480.54 97.084 479.191 99.292C477.842 101.5 477.167 104.199 477.167 107.388V136H470.543ZM530.227 137.012C528.019 137.012 526.24 136.613 524.891 135.816C523.603 135.019 522.683 133.915 522.131 132.504C521.64 131.093 521.395 129.529 521.395 127.812V93.22H514.863V87.608H521.395L522.683 74.176H528.019V87.608H537.311V93.22H528.019V126.616C528.019 128.272 528.233 129.499 528.663 130.296C529.153 131.093 530.196 131.492 531.791 131.492H537.311V135.724C536.697 135.969 535.961 136.184 535.103 136.368C534.305 136.613 533.477 136.767 532.619 136.828C531.76 136.951 530.963 137.012 530.227 137.012ZM544.308 77.672V69.484H550.932V77.672H544.308ZM544.308 136V87.608H550.932V136H544.308ZM563.802 136V93.22H556.626V87.608H563.802V79.144C563.802 77.4267 564.048 75.7707 564.538 74.176C565.09 72.5813 566.072 71.2933 567.482 70.312C568.893 69.2693 570.886 68.748 573.462 68.748C574.26 68.748 575.057 68.7787 575.854 68.84C576.652 68.9013 577.388 68.9933 578.062 69.116C578.798 69.2387 579.442 69.4227 579.994 69.668V74.36H575.118C573.524 74.36 572.328 74.7587 571.53 75.556C570.794 76.3533 570.426 77.5187 570.426 79.052V87.608H579.994V93.22H570.426V136H563.802ZM586.71 77.672V69.484H593.334V77.672H586.71ZM586.71 136V87.608H593.334V136H586.71ZM623.183 137.104C618.583 137.104 614.689 136.215 611.499 134.436C608.31 132.596 605.887 129.805 604.231 126.064C602.575 122.323 601.747 117.569 601.747 111.804C601.747 105.977 602.575 101.224 604.231 97.544C605.887 93.8027 608.341 91.0427 611.591 89.264C614.842 87.424 618.89 86.504 623.735 86.504C628.213 86.504 631.954 87.3933 634.959 89.172C638.026 90.9507 640.326 93.5267 641.859 96.9C643.454 100.273 644.251 104.413 644.251 109.32V113.552H608.647C608.647 117.784 609.23 121.249 610.395 123.948C611.561 126.647 613.217 128.609 615.363 129.836C617.571 131.001 620.27 131.584 623.459 131.584C625.79 131.584 627.814 131.277 629.531 130.664C631.31 130.051 632.751 129.223 633.855 128.18C635.021 127.076 635.91 125.788 636.523 124.316C637.137 122.844 637.443 121.311 637.443 119.716H643.883C643.883 122.108 643.423 124.377 642.503 126.524C641.645 128.609 640.357 130.449 638.639 132.044C636.983 133.639 634.867 134.896 632.291 135.816C629.715 136.675 626.679 137.104 623.183 137.104ZM608.647 108.308H637.351C637.351 105.241 636.983 102.696 636.247 100.672C635.573 98.5867 634.622 96.9307 633.395 95.704C632.169 94.416 630.697 93.496 628.979 92.944C627.323 92.392 625.453 92.116 623.367 92.116C620.362 92.116 617.786 92.6987 615.639 93.864C613.493 94.968 611.837 96.716 610.671 99.108C609.567 101.5 608.893 104.567 608.647 108.308ZM652.441 136V87.608H657.777L658.329 95.704H658.973C659.402 94.232 660.046 92.7907 660.905 91.38C661.763 89.9693 662.929 88.804 664.401 87.884C665.873 86.964 667.743 86.504 670.013 86.504C670.871 86.504 671.669 86.596 672.405 86.78C673.141 86.9027 673.693 87.0253 674.061 87.148V93.22H671.209C669.062 93.22 667.222 93.6187 665.689 94.416C664.155 95.152 662.898 96.2253 661.917 97.636C660.935 99.0467 660.199 100.672 659.709 102.512C659.279 104.291 659.065 106.161 659.065 108.124V136H652.441Z" fill="currentColor"/> </g> </g> </svg>
      <p class="hero-line">
        <span class="sr-only">Antimicrobial peptide predictor</span>
        <span class="hero-scramble" id="heroScramble" aria-hidden="true"
              data-text="Antimicrobial peptide predictor"></span>
      </p>
    </div>

  </header>

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
  </main>

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


  <footer class="step-2">
    <div>
      <div class="usage-map-title">Where AMPidentifier is being used</div>
      <div id="usageMap"></div>
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
  </footer>
"""

CSS = """
  /* A marca abre a pagina: leque em teal e a palavra na tinta do modo.
     O SVG entra em linha e nao por <img> justamente por isso: dentro de
     uma imagem o CSS nao alcanca o caminho da palavra, e a alternativa
     seria inverter a marca inteira, o que devolveria o complementar do
     teal em vez da versao para fundo escuro. */
  .hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-16);
    margin-bottom: var(--space-48);
  }

  /* A frase se resolve a partir do codigo de uma letra dos aminoacidos, e
     nao de simbolos quaisquer: a materia da ferramenta e a sequencia, e
     usar o alfabeto dela liga o efeito ao assunto em vez de decorar.

     Mono e obrigatorio aqui, nao preferencia: cada glifo tem o mesmo
     avanco, entao a linha nao muda de largura enquanto as letras giram.
     Em fonte proporcional a frase pularia a cada quadro.

     O texto final tambem existe em .sr-only, e o span animado e
     aria-hidden: um leitor de tela lendo o embaralhamento leria lixo. */
  .hero-line { margin: 0; min-height: var(--space-20); }

  .hero-scramble {
    font-family: var(--font-mono);
    font-size: var(--text-13);
    letter-spacing: var(--tracking-wide);
    color: var(--muted);
    white-space: pre;
  }

  /* O ciclo apaga em fade e recomeca. So opacidade anima, e a saida e
     mais curta que a entrada, porque quem ja leu a frase nao precisa
     assistir ela sair. */
  .hero-scramble {
    transition: opacity var(--duration-5) var(--ease-out-soft);
  }

  .hero-scramble.is-out { opacity: 0; }

  /* a letra ja resolvida vai para a tinta cheia e as que ainda giram
     ficam em --muted: a frase se forma da esquerda para a direita e da
     para ver onde ela esta */
  .hero-scramble .done { color: var(--text); }
  .hero-fan { display: block; height: var(--space-80); width: auto; color: var(--text); }

  @media (max-width: 768px) {
    .hero-fan { height: var(--space-56); }
  }
"""

JS = """/* ---------- a frase se resolve a partir dos aminoacidos ----------
   O alfabeto do embaralhamento e o codigo de uma letra dos vinte
   aminoacidos, e nao simbolos quaisquer: a materia da ferramenta e a
   sequencia, entao a frase nasce do alfabeto dela. Nao e enfeite com
   tema, e o tema virando enfeite.

   Mono e obrigatorio, nao preferencia: cada glifo tem o mesmo avanco,
   entao a linha nao muda de largura enquanto as letras giram. Em fonte
   proporcional a frase pularia a cada quadro.

   O ciclo: resolve da esquerda para a direita, segura, apaga em fade,
   espera e recomeca. Ele para quando a aba sai de vista, porque animar
   o que ninguem esta vendo e gastar bateria, e nao roda nenhuma vez sob
   movimento reduzido: la a frase simplesmente esta escrita.

   Os numeros abaixo sao de tempo, em milissegundos, e nenhum e lido por
   folha de estilo, entao nenhum vira token. */

(function () {
  var AMINO = 'ACDEFGHIKLMNPQRSTVWY';
  var TICK = 45;        /* quanto tempo cada giro fica na tela */
  var PER_CHAR = 2;     /* giros antes de uma letra assentar */
  var HOLD = 2600;      /* frase inteira parada, para ser lida */
  var GAP = 500;        /* escuro entre um ciclo e o proximo */

  var host = document.getElementById('heroScramble');
  if (!host) return;
  var text = host.dataset.text || '';
  var still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (still) {
    host.textContent = text;
    return;
  }

  var timer = null;

  function rand() {
    return AMINO[Math.floor(Math.random() * AMINO.length)];
  }

  /* uma passada: `resolvidas` letras ja assentadas, o resto girando */
  function frame(resolvidas) {
    var out = '';
    for (var i = 0; i < text.length; i++) {
      var ch = text[i];
      if (ch === ' ') { out += ' '; continue; }
      if (i < resolvidas) out += '<span class="done">' + ch + '</span>';
      else out += rand();
    }
    host.innerHTML = out;
  }

  function run() {
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

  function stop() {
    window.clearInterval(timer);
    window.clearTimeout(timer);
    timer = null;
  }

  /* aba escondida nao anima: setInterval continua correndo em segundo
     plano, ao contrario de requestAnimationFrame, entao aqui a parada e
     explicita */
  document.addEventListener('visibilitychange', function () {
    if (document.hidden) { stop(); host.textContent = text; }
    else if (!timer) run();
  });

  run();
})();

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

# ---------------------------------------------------------------------------
# Versao em portugues.
#
# Nao se traduz: nome da marca, sigla de metrica (AUC-ROC, MCC), nome de
# modelo (Random forest, XGBoost), nome de instituicao, numero de registro
# no INPI, a citacao e o codigo FASTA. Sao registros, e traduzir registro
# quebra o casamento com a fonte que o declara.
# ---------------------------------------------------------------------------

BODY_PAIRS = [
    ('<span class="sr-only">Antimicrobial peptide predictor</span>',
     '<span class="sr-only">Preditor de peptideos antimicrobianos</span>'),
    ('data-text="Antimicrobial peptide predictor"',
     'data-text="Preditor de peptideos antimicrobianos"'),
    ('>AMPidentifier, antimicrobial peptide prediction</h1>',
     '>AMPidentifier, predi&ccedil;&atilde;o de pept&iacute;deos antimicrobianos</h1>'),
    ('is a toolkit for antimicrobial peptide prediction using ensemble machine learning.',
     '&eacute; um conjunto de ferramentas para predi&ccedil;&atilde;o de pept&iacute;deos antimicrobianos com aprendizado de m&aacute;quina em ensemble.'),
    ('>For <a href="https://pypi.org/project/ampidentifier/" target="_blank">PyPI</a>:</span>',
     '>No <a href="https://pypi.org/project/ampidentifier/" target="_blank">PyPI</a>:</span>'),
    ('>For terminal use:</span>', '>Para uso no terminal:</span>'),
    ('>CLI version</a>', '>vers&atilde;o de linha de comando</a>'),
    ('>In testing:</span>', '>Em teste:</span>'),
    ('>what is coming next</a>', '>o que vem por a&iacute;</a>'),
    ('>Previous layout:</span>', '>Layout anterior:</span>'),
    ('>legacy version</a>', '>vers&atilde;o legada</a>'),
    ('>Benchmark, voting ensemble (RF + SVM + GB + XGB + LGBM)</div>',
     '>Benchmark, ensemble por vota&ccedil;&atilde;o (RF + SVM + GB + XGB + LGBM)</div>'),
    ('>Sensitivity</span>', '>Sensibilidade</span>'),
    ('>Specificity</span>', '>Especificidade</span>'),
    ('>Usage</div>', '>Uso</div>'),
    ('>Sequences classified</span>', '>Sequ&ecirc;ncias lidas</span>'),
    ('>Unique users</span>', '>Usu&aacute;rios &uacute;nicos</span>'),
    ('>Prediction runs</span>', '>Execu&ccedil;&otilde;es</span>'),
    ('>Descriptors</span>', '>Descritores</span>'),
    ('>FASTA sequences</label>', '>Sequ&ecirc;ncias FASTA</label>'),
    ('>Voting ensemble</option>', '>Ensemble por vota&ccedil;&atilde;o</option>'),
    ('>Run</button>', '>Executar</button>'),
    ('>Clear</button>', '>Limpar</button>'),
    ('>Load example</button>', '>Carregar exemplo</button>'),
    ('>Upload .fasta</button>', '>Enviar .fasta</button>'),
    ('>Find AMPidentifier useful?</div>', '>Achou o AMPidentifier &uacute;til?</div>'),
    ('>Copy link</button>', '>Copiar link</button>'),
    ('>Share by email</button>', '>Compartilhar por email</button>'),
    ('>Recipient email</span>', '>Email do destinat&aacute;rio</span>'),
    ('>Send</button>', '>Enviar</button>'),
    ('>Where AMPidentifier is being used</div>', '>Onde o AMPidentifier est&aacute; sendo usado</div>'),
    ('This tool is officially registered with the',
     'Esta ferramenta est&aacute; registrada no'),
    ('(Brazilian National Institute of Industrial Property), Registration No.',
     '(Instituto Nacional da Propriedade Industrial), sob o n&uacute;mero'),
    ('. It is a property of the', '. &Eacute; propriedade da'),
    ('Your data is encrypted during transfer (HTTPS/TLS) and never shared. Sequences are not stored.',
     'Seus dados s&atilde;o criptografados na transfer&ecirc;ncia (HTTPS/TLS) e nunca compartilhados. Sequ&ecirc;ncias n&atilde;o s&atilde;o armazenadas.'),
    ('>Developer: <a href="mailto:', '>Desenvolvedor: <a href="mailto:'),
    ('>Report issue or suggestion</button>', '>Relatar problema ou sugest&atilde;o</button>'),
]

CHANGELOG_PAIRS = [
]

BODY_PT = BODY
for _a, _b in BODY_PAIRS + CHANGELOG_PAIRS:
    if _a in BODY_PT:
        BODY_PT = BODY_PT.replace(_a, _b, 1)

# os dois links da caixa de instalacao apontam para a rota em portugues
BODY_PT = (BODY_PT
           .replace('<a href="/beta">', '<a href="/pt/beta">')
           .replace('<a href="/legacy">', '<a href="/legacy">'))


PAGE = page(
    title='AMPidentifier | Antimicrobial Peptide Prediction Tool',
    description=('AMPidentifier is a free web tool for antimicrobial peptide (AMP) '
                 'prediction using machine learning ensemble models. Submit FASTA '
                 'sequences and classify AMPs in seconds.'),
    path='/',
    body=BODY,
    css=CSS,
    js=JS,
)

PAGE_PT = page(
    title='AMPidentifier | Predi\u00e7\u00e3o de pept\u00eddeos antimicrobianos',
    description=('O AMPidentifier \u00e9 uma ferramenta web gratuita para predi\u00e7\u00e3o de '
                 'pept\u00eddeos antimicrobianos com modelos de aprendizado de m\u00e1quina em '
                 'ensemble. Envie sequ\u00eancias FASTA e classifique em segundos.'),
    path='/',
    body=BODY_PT,
    css=CSS,
    js=JS,
    lang='pt',
)
