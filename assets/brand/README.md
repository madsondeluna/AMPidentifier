# AMPidentifier brand assets

Source file: [Figma](https://www.figma.com/design/luDggo2jYMDtnDHVBAAYBZ/AMPidentifier-logo-explorations)

Three marks, all built from the same vocabulary as the AMPcraft mark: flat dots, one colour, Archivo wordmark with `AMP` in SemiBold and `identifier` in Light at the same size, letter-spacing -2%, exported as outlines. No font is needed to render these files.

| Variant | What it shows | Use |
|---|---|---|
| `vote` | Five models fanning upward over one call | The soft-voting ensemble, wide and short |
| `fan` | The same five votes fanning sideways, away from the wordmark | Same idea, taller silhouette, reads better next to long text |
| `motif` | Three aligned sequences with the deciding column enlarged | The positional feature set that separates this version from the beta pipeline |

## Colour

Twelve colours, the same set AMPcraft uses, so the two projects read as one family. `mid` is the mark on light backgrounds, `bright` is the mark on dark backgrounds. The wordmark is always ink `#12181C` on light and white on dark.

| Colour | mid | bright |
|---|---|---|
| teal (primary) | `#0E6E66` | `#3FB0A4` |
| emerald | `#12805A` | `#5EA98F` |
| sea | `#0F7C86` | `#5CA6AD` |
| cobalt | `#1D5FBF` | `#6592D3` |
| indigo | `#3A46B5` | `#7981CD` |
| violet | `#6B3FB0` | `#9A7CC9` |
| magenta | `#A62C74` | `#C270A0` |
| crimson | `#B32B3E` | `#CB6F7C` |
| rust | `#B4551F` | `#CC8B67` |
| amber | `#B0790F` | `#C9A45C` |
| olive | `#6E8B22` | `#9CB069` |
| slate | `#46545E` | `#818B92` |

Every `bright` is `mid` mixed 32% toward white, except teal, which keeps the tone AMPcraft already uses.

Neutrals: ink `#12181C`, black `#0A0D0F`, gray mark `#6B7376`, gray word `#2E3234`. The four-tone grayscale set is `#3B4143`, `#6B7376`, `#9AA1A4`, `#C2C7C9`.

## Colour combinations

Every mark has parts that can be coloured apart, and in each case the split says something true about the pipeline.

Twenty combinations, each in every form and every variant.

| Combination | In `vote` and `fan` | In `motif` |
|---|---|---|
| Twelve duos: `duo-teal-amber`, `duo-teal-crimson`, `duo-teal-magenta`, `duo-teal-cobalt`, `duo-indigo-amber`, `duo-slate-teal`, `duo-cobalt-crimson`, `duo-emerald-amber`, `duo-violet-amber`, `duo-rust-teal`, `duo-olive-crimson`, `duo-sea-magenta` | Models in the first colour, the final call in the second | Field residues in the first colour, the deciding column in the second |
| Six multis: `multi-cool`, `multi-warm`, `multi-mixed`, `multi-spectrum`, `multi-earth`, `multi-ocean` | One colour per model, five models, five colours, the call in the first | Field in the first colour, the three motif residues each in their own |
| `grayscale-multi`, `grayscale-duo` | Four grays and two grays, the print fallbacks | Same |

Every combination also exists with the `-dark` suffix, retoned for dark backgrounds with a white wordmark.

## Forms and treatments

| Form | Content |
|---|---|
| `lockup` | mark left, AMPidentifier right, generous gap |
| `compact` | the same, mark larger and the gap tightened |
| `stacked` | mark above, AMPidentifier below |
| `symbol` | mark alone |

| Treatment | Mark | Word | For |
|---|---|---|---|
| `color` | teal mid | ink | light backgrounds |
| `color-dark` | teal bright | white | dark backgrounds |
| `black` | `#0A0D0F` | `#0A0D0F` | single-colour print |
| `white` | white | white | over photographs and solid colour |
| `gray` | gray mark | gray word | documents where colour would compete |

Every SVG has a matching PNG with a transparent background. The teal and neutral set exports at 4x; `colors/` and `combos/` export at 3x, which still puts a lockup past 1900 px wide and keeps the folder from doubling in size. SVG carries no background either.

## Files

```
assets/brand/
  <variant>/            teal and the neutrals, svg and png, plus the cards with a background
  <variant>/colors/     the eleven non-teal colours, four forms, light and dark
  <variant>/combos/     the twenty combinations, four forms, light and dark
  _base/                the twelve exports from Figma, the only handwritten input
  generate.py           rebuilds every file above from _base
  preview.py            rebuilds the nine contact sheets
```

File names read `ampidentifier-<form>-<variant>-<treatment>.<ext>`, for example `ampidentifier-lockup-vote-color.svg` or `ampidentifier-symbol-motif-multi-cool-dark.svg`.

Cards that carry a background: `ampidentifier-lockup-<variant>-on-white|on-teal|on-ink.png` at 2400x800, and `ampidentifier-icon-<variant>-on-white|on-teal|on-ink.png` at 1024x1024, the second set sized for favicons and avatars.

Contact sheets, three per variant, nine in total: `preview-<variant>-forms.png`, `preview-<variant>-colors.png`, `preview-<variant>-combos.png`.

## Regenerating

Each base in `_base/` carries two colours: `#0E6E66` on every mark shape and `#12181C` on the wordmark. The mark shapes appear in a known order, which is what lets a combination colour one element at a time: in `vote` and `fan` the shapes run Model 1, Vote 1, Model 2, Vote 2, through Model 5, Vote 5, then Call; in `motif` the eighteen residues run row by row, with the deciding column at index 3 of each row.

```bash
cd assets/brand
python3 generate.py
python3 preview.py
```

Requires `rsvg-convert` and ImageMagick. Adding a colour or a combination means editing the lists at the top of `generate.py`. Changing the geometry means editing the Figma file and replacing `_base/`.

## Rules

`lockup` is the default. `stacked` is for square and narrow spaces. `symbol` is for avatars, favicons, and anywhere the name is already present.

Clear space is half the height of the mark on every side. Below 120 px wide, drop the lockup and use the symbol.

Do not place a `color` file on a dark background: use `color-dark`. Do not recolour a file by hand.

## In use

| Where | File |
|---|---|
| Repository README | `fan/colors/ampidentifier-compact-fan-slate.svg` |
| Web app header, `webapp/app.py` and `webapp/page_beta.py` | the same file, copied to `webapp/img/logo.svg` |

The site serves its own copy, so after regenerating run:

```bash
cp fan/colors/ampidentifier-compact-fan-slate.svg ../../webapp/img/logo.svg
```

The previous kit lives in `imgs/ampidentifier-brand-kit/` and is untouched, along with `webapp/img/logo.png`. Nothing there is referenced any more.
