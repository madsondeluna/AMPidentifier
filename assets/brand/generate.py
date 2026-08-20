"""Generate every AMPidentifier brand file from the six base SVGs exported from Figma.

Source of truth: https://www.figma.com/design/luDggo2jYMDtnDHVBAAYBZ/AMPidentifier-logo-explorations
Each base in _base/ carries two colours: #0E6E66 on every mark shape and #12181C on the
wordmark. The mark shapes appear in a known order, which is what lets the combination
files colour one element at a time:

  vote   Model 1, Vote 1, Model 2, Vote 2, ... Model 5, Vote 5, Call   (11 shapes)
  motif  18 residues in row-major order; the deciding column is index 3 of each row

Every SVG has a matching PNG at 4x. Requires rsvg-convert and ImageMagick.
Run: python3 generate.py
"""
import os, re, shutil, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, '_base')
MARK, WORD = '#0E6E66', '#12181C'
INK, WHITE, BLACK = '#12181C', '#FFFFFF', '#0A0D0F'
GRAY_MARK, GRAY_WORD = '#6B7376', '#2E3234'
GRAYS = ['#3B4143', '#6B7376', '#9AA1A4', '#C2C7C9']

PALETTE = {
    'teal':    ('#0E6E66', '#3FB0A4'),
    'emerald': ('#12805A', None),
    'sea':     ('#0F7C86', None),
    'cobalt':  ('#1D5FBF', None),
    'indigo':  ('#3A46B5', None),
    'violet':  ('#6B3FB0', None),
    'magenta': ('#A62C74', None),
    'crimson': ('#B32B3E', None),
    'rust':    ('#B4551F', None),
    'amber':   ('#B0790F', None),
    'olive':   ('#6E8B22', None),
    'slate':   ('#46545E', None),
}
VARIANTS = ['vote', 'fan', 'motif']
FORMS = ['lockup', 'compact', 'stacked', 'symbol']
TREATMENTS = ['color', 'color-dark', 'black', 'white', 'gray']

DUOS = [
    ('duo-teal-amber',      ['teal', 'amber']),
    ('duo-teal-crimson',    ['teal', 'crimson']),
    ('duo-teal-magenta',    ['teal', 'magenta']),
    ('duo-teal-cobalt',     ['teal', 'cobalt']),
    ('duo-indigo-amber',    ['indigo', 'amber']),
    ('duo-slate-teal',      ['slate', 'teal']),
    ('duo-cobalt-crimson',  ['cobalt', 'crimson']),
    ('duo-emerald-amber',   ['emerald', 'amber']),
    ('duo-violet-amber',    ['violet', 'amber']),
    ('duo-rust-teal',       ['rust', 'teal']),
    ('duo-olive-crimson',   ['olive', 'crimson']),
    ('duo-sea-magenta',     ['sea', 'magenta']),
]
MULTIS = [
    ('multi-cool',     ['teal', 'sea', 'cobalt', 'indigo', 'violet']),
    ('multi-warm',     ['amber', 'rust', 'crimson', 'magenta', 'violet']),
    ('multi-mixed',    ['teal', 'amber', 'crimson', 'indigo', 'olive']),
    ('multi-spectrum', ['teal', 'cobalt', 'violet', 'magenta', 'crimson']),
    ('multi-earth',    ['olive', 'amber', 'rust', 'crimson', 'slate']),
    ('multi-ocean',    ['emerald', 'teal', 'sea', 'cobalt', 'indigo']),
]


def hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def mix(a, b, t):
    ra, rb = hex2rgb(a), hex2rgb(b)
    return '#%02X%02X%02X' % tuple(int(round(ra[i] + (rb[i] - ra[i]) * t)) for i in range(3))


def tone(name, dark):
    mid, bright = PALETTE[name]
    return (bright or mix(mid, WHITE, 0.32)) if dark else mid


def base_svg(form, variant):
    s = open(os.path.join(SRC, 'ampidentifier-%s-%s.svg' % (form, variant)), encoding='utf-8').read()
    return re.sub(r'<rect width="[0-9.]+" height="[0-9.]+" fill="#E5E5E5"/>', '', s)


def paint(svg, per_shape, word):
    """per_shape: callable index -> colour, applied to the mark fills in document order."""
    idx = [0]

    def sub(_m):
        k = idx[0]
        idx[0] += 1
        return 'fill="%s"' % per_shape(k)

    return re.sub(r'fill="%s"' % MARK, sub, svg).replace(WORD, word)


def flat(colour):
    return lambda k: colour


def duo_map(variant, first, second):
    """First colour is the field, second is the element that carries the decision."""
    if variant in ('vote', 'fan'):
        return lambda k: second if k == 10 else first        # Call is the last shape
    return lambda k: second if k % 6 == 3 else first         # the deciding column


def multi_map(variant, colours):
    if variant in ('vote', 'fan'):
        def m(k):
            if k == 10:
                return colours[0]
            return colours[(k // 2) % len(colours)]          # one colour per model
        return m

    def m(k):
        if k % 6 == 3:
            return colours[(k // 6) + 1 if (k // 6) + 1 < len(colours) else 0]
        return colours[0]
    return m


def png_of(path, scale=4):
    w = float(re.search(r'width="([0-9.]+)"', open(path, encoding='utf-8').read()).group(1))
    out = path[:-4] + '.png'
    subprocess.run(['rsvg-convert', '-w', str(int(round(w * scale))), path, '-o', out], check=True)
    return out


def write(path, svg, scale=4):
    open(path, 'w', encoding='utf-8').write(svg)
    return [path, png_of(path, scale)]


def card(src_svg, out_png, cw, ch, bg, inner):
    tmp = out_png + '.tmp.png'
    subprocess.run(['rsvg-convert', '-w', str(inner), src_svg, '-o', tmp], check=True)
    subprocess.run(['magick', '-size', '%dx%d' % (cw, ch), 'xc:' + bg,
                    tmp, '-gravity', 'center', '-composite', out_png], check=True)
    os.remove(tmp)


made = []
for variant in VARIANTS:
    vdir = os.path.join(BASE, variant)
    if os.path.isdir(vdir):
        shutil.rmtree(vdir)
    for sub_dir in ('', 'colors', 'combos'):
        os.makedirs(os.path.join(vdir, sub_dir), exist_ok=True)
    for form in FORMS:
        svg = base_svg(form, variant)
        plans = {
            'color':      (tone('teal', False), INK),
            'color-dark': (tone('teal', True), WHITE),
            'black':      (BLACK, BLACK),
            'white':      (WHITE, WHITE),
            'gray':       (GRAY_MARK, GRAY_WORD),
        }
        for t in TREATMENTS:
            c, w = plans[t]
            made += write(os.path.join(vdir, 'ampidentifier-%s-%s-%s.svg' % (form, variant, t)),
                          paint(svg, flat(c), w))
        for name in PALETTE:
            if name == 'teal':
                continue
            for dark in (False, True):
                c = tone(name, dark)
                w = WHITE if dark else INK
                made += write(os.path.join(vdir, 'colors', 'ampidentifier-%s-%s-%s%s.svg'
                                           % (form, variant, name, '-dark' if dark else '')),
                              paint(svg, flat(c), w), scale=3)
        for cname, names in DUOS:
            for dark in (False, True):
                a, b = tone(names[0], dark), tone(names[1], dark)
                w = WHITE if dark else INK
                made += write(os.path.join(vdir, 'combos', 'ampidentifier-%s-%s-%s%s.svg'
                                           % (form, variant, cname, '-dark' if dark else '')),
                              paint(svg, duo_map(variant, a, b), w), scale=3)
        for cname, names in MULTIS:
            for dark in (False, True):
                cols = [tone(n, dark) for n in names]
                w = WHITE if dark else INK
                made += write(os.path.join(vdir, 'combos', 'ampidentifier-%s-%s-%s%s.svg'
                                           % (form, variant, cname, '-dark' if dark else '')),
                              paint(svg, multi_map(variant, cols), w), scale=3)
        for dark in (False, True):
            cols = list(reversed(GRAYS)) if dark else GRAYS
            w = WHITE if dark else GRAY_WORD
            made += write(os.path.join(vdir, 'combos', 'ampidentifier-%s-%s-grayscale-multi%s.svg'
                                       % (form, variant, '-dark' if dark else '')),
                          paint(svg, multi_map(variant, cols), w), scale=3)
            a, b = (GRAYS[3], GRAYS[1]) if dark else (GRAYS[2], GRAYS[0])
            made += write(os.path.join(vdir, 'combos', 'ampidentifier-%s-%s-grayscale-duo%s.svg'
                                       % (form, variant, '-dark' if dark else '')),
                          paint(svg, duo_map(variant, a, b), w), scale=3)
    teal = tone('teal', False)
    lk = os.path.join(vdir, 'ampidentifier-lockup-%s-' % variant)
    sy = os.path.join(vdir, 'ampidentifier-symbol-%s-' % variant)
    card(lk + 'color.svg', lk + 'on-white.png', 2400, 800, WHITE, 1560)
    card(lk + 'white.svg', lk + 'on-teal.png', 2400, 800, teal, 1560)
    card(lk + 'color-dark.svg', lk + 'on-ink.png', 2400, 800, INK, 1560)
    card(sy + 'color.svg', os.path.join(vdir, 'ampidentifier-icon-%s-on-white.png' % variant), 1024, 1024, WHITE, 520)
    card(sy + 'white.svg', os.path.join(vdir, 'ampidentifier-icon-%s-on-teal.png' % variant), 1024, 1024, teal, 520)
    card(sy + 'color-dark.svg', os.path.join(vdir, 'ampidentifier-icon-%s-on-ink.png' % variant), 1024, 1024, INK, 520)

    # banners on a white card, one per colour, for READMEs and slides where the
    # page behind is dark and a transparent lockup would disappear
    for name in PALETTE:
        src = (os.path.join(vdir, 'ampidentifier-compact-%s-color.svg' % variant) if name == 'teal'
               else os.path.join(vdir, 'colors', 'ampidentifier-compact-%s-%s.svg' % (variant, name)))
        out = os.path.join(vdir, 'ampidentifier-compact-%s-%s-on-white.png' % (variant, name))
        card(src, out, 2400, 800, WHITE, 1560)
        made.append(out)

for v in VARIANTS:
    d = os.path.join(BASE, v)
    root = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
    print('%-6s raiz %3d | colors %3d | combos %3d'
          % (v, len(root), len(os.listdir(os.path.join(d, 'colors'))), len(os.listdir(os.path.join(d, 'combos')))))
print('total gerado:', len(made))
