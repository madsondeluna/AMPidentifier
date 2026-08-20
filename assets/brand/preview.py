"""Build the AMPidentifier contact sheets, three per variant.

preview-<variant>-forms.png    every form in every treatment
preview-<variant>-colors.png   every colour, light and dark, plus the symbols in a row
preview-<variant>-combos.png   every combination, light and dark

Run after generate.py. Requires rsvg-convert and ImageMagick.
"""
import os, subprocess, tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
INK, WHITE, PAPER = '#12181C', '#FFFFFF', '#EFEFED'
VARIANTS = ['vote', 'fan', 'motif']
FORMS = ['lockup', 'compact', 'stacked', 'symbol']
COLORS = ['teal', 'emerald', 'sea', 'cobalt', 'indigo', 'violet',
          'magenta', 'crimson', 'rust', 'amber', 'olive', 'slate']
COMBOS = ['duo-teal-amber', 'duo-teal-crimson', 'duo-teal-magenta', 'duo-teal-cobalt',
          'duo-indigo-amber', 'duo-slate-teal', 'duo-cobalt-crimson', 'duo-emerald-amber',
          'duo-violet-amber', 'duo-rust-teal', 'duo-olive-crimson', 'duo-sea-magenta',
          'multi-cool', 'multi-warm', 'multi-mixed', 'multi-spectrum', 'multi-earth',
          'multi-ocean', 'grayscale-multi', 'grayscale-duo']
n = [0]


def raster(svg, width, tmp):
    n[0] += 1
    out = os.path.join(tmp, 'r%d.png' % n[0])
    subprocess.run(['rsvg-convert', '-w', str(width), svg, '-o', out], check=True)
    return out


def strip(files, bg, tile, geom, out):
    subprocess.run(['magick', 'montage'] + files +
                   ['-tile', tile, '-geometry', geom, '-background', bg, out], check=True)
    return out


def stack(rows, out, width=1500):
    subprocess.run(['magick'] + rows + ['-background', PAPER, '-gravity', 'center',
                                        '-append', '-resize', '%dx' % width, out], check=True)
    subprocess.run(['magick', out, '-strip', '-colors', '128', out], check=True)
    return out


with tempfile.TemporaryDirectory() as tmp:
    for v in VARIANTS:
        d = os.path.join(BASE, v)

        def f(name):
            return os.path.join(d, name)

        def sub(folder, name):
            return os.path.join(d, folder, name)

        def prep(paths, w=640):
            return [raster(p, w, tmp) for p in paths]

        rows = []
        for treat, bg in (('color', WHITE), ('black', WHITE), ('gray', WHITE),
                          ('color-dark', INK), ('white', INK)):
            files = [f('ampidentifier-%s-%s-%s.svg' % (form, v, treat)) for form in FORMS]
            rows.append(strip(prep(files), bg, '3x', '400x180+16+16',
                              os.path.join(tmp, '%s-forms-%s.png' % (v, treat))))
        print('escrito:', os.path.basename(stack(rows, os.path.join(BASE, 'preview-%s-forms.png' % v))))

        rows = []
        light = [f('ampidentifier-lockup-%s-color.svg' % v) if c == 'teal'
                 else sub('colors', 'ampidentifier-lockup-%s-%s.svg' % (v, c)) for c in COLORS]
        dark = [f('ampidentifier-lockup-%s-color-dark.svg' % v) if c == 'teal'
                else sub('colors', 'ampidentifier-lockup-%s-%s-dark.svg' % (v, c)) for c in COLORS]
        sym = [f('ampidentifier-symbol-%s-color.svg' % v) if c == 'teal'
               else sub('colors', 'ampidentifier-symbol-%s-%s.svg' % (v, c)) for c in COLORS]
        rows.append(strip(prep(light), WHITE, '4x', '340x120+14+14', os.path.join(tmp, v + '-cl.png')))
        rows.append(strip(prep(dark), INK, '4x', '340x120+14+14', os.path.join(tmp, v + '-cd.png')))
        rows.append(strip(prep(sym), WHITE, '12x', '110x120+10+10', os.path.join(tmp, v + '-cs.png')))
        print('escrito:', os.path.basename(stack(rows, os.path.join(BASE, 'preview-%s-colors.png' % v))))

        rows = []
        light = [sub('combos', 'ampidentifier-lockup-%s-%s.svg' % (v, c)) for c in COMBOS]
        dark = [sub('combos', 'ampidentifier-lockup-%s-%s-dark.svg' % (v, c)) for c in COMBOS]
        rows.append(strip(prep(light), WHITE, '3x', '400x140+16+16', os.path.join(tmp, v + '-kl.png')))
        rows.append(strip(prep(dark), INK, '3x', '400x140+16+16', os.path.join(tmp, v + '-kd.png')))
        print('escrito:', os.path.basename(stack(rows, os.path.join(BASE, 'preview-%s-combos.png' % v))))
