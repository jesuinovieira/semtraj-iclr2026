# HTML/CSS Research Poster — Reusable Knowledge

## Fixed-sheet architecture

Use a fixed-size `#poster` div matching the target paper dimensions. The body has a neutral background so the poster appears as a centered sheet in the browser — matching how it will look printed.

```css
#poster {
  width: <paper-width>;
  height: <paper-height>;
  overflow: hidden;
}
body {
  background: #bbb;
  display: flex;
  justify-content: center;
}
```

The `overflow: hidden` is critical — content that exceeds the sheet height is silently clipped. Always verify the footer remains visible after content changes.

## Three-column flowing layout

Use CSS Grid for the top-level structure (header, columns, footer) and flexbox columns for the content within each column. This avoids the "rigid same-size boxes" problem that a full grid creates.

```css
#poster {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  grid-template-rows: auto 1fr auto;
  grid-template-areas:
    "header header header"
    "col1   col2   col3"
    "footer footer footer";
}
.col {
  display: flex;
  flex-direction: column;
  gap: 5mm;
}
```

Content within each column stacks naturally with organic heights — sections take only the space they need. This makes the poster feel dynamic rather than rigid.

Reading order is **top-to-bottom per column** (standard for scientific posters). Section headings visually separate topics within a column.

## Balancing column heights

The biggest layout challenge is avoiding large empty spaces at the bottom of columns. Strategies:

1. **Redistribute content** between columns until heights roughly match
2. **Move a section** to a short column to fill it
3. **Adjust figure/card padding** — figures and cards are the most compressible elements

Check the rendered poster after any content change — a few mm of padding difference can cascade into a visible gap at the bottom of a column.

## Printing to PDF

Open the poster HTML in Chrome → `Cmd+P`:

1. **More settings** → expand
2. **Paper size** → set to the target size (may need "Custom")
3. **Margins** → None
4. **Background graphics** → checked
5. **Scale** → 100%

If Chrome shows multiple pages, the paper size is wrong (likely defaulting to Letter/A4).

## SVG figures

Use `<img>` tags (not `<object>`) for poster SVGs. `<img>` is more reliable for print/PDF export — `<object>` can cause blank figures when printing.

```html
<div class="figure">
  <img src="../figures/diagram.svg" alt="...">
</div>
```

## Reusing a presentation's design system

When a poster accompanies a slide presentation, inherit the same theme — colors, fonts, and component patterns (cards, callouts, tags). This ensures visual consistency. Pick the light mode variant since posters are printed on white paper.

The poster can introduce one or two unique elements (e.g., an inverted-color card for the conclusion) to differentiate it from the slides without breaking the design language.

## Spacing budget

With a fixed sheet height, space is tight. Approximate breakdown:

| Element       | Height  |
|---------------|---------|
| Padding (top+bottom) | ~24mm |
| Header        | ~40mm |
| Column gap    | ~8mm  |
| Footer        | ~22mm |
| **Available for columns** | **remainder** |

When adjusting content, reduce in this order: padding inside cards → gaps between items → figure padding → font sizes (last resort).

## Background grid pattern

A subtle grid pattern on the poster sheet (not on `body`) adds texture without competing with content:

```css
background-image:
  linear-gradient(rgba(40,60,120,0.04) 1px, transparent 1px),
  linear-gradient(90deg, rgba(40,60,120,0.04) 1px, transparent 1px);
background-size: 20mm 20mm;
```

## Section titles: assertion-evidence style

Use descriptive, forward-looking titles that convey the takeaway — not generic labels like "The Problem" or "Conclusion". A passerby skimming the poster should grasp the story from the headings alone.

**Do:** "Patients show more erratic, constricted trajectories"
**Don't:** "Result 1: Neurodegenerative"

**Do:** "Cumulative embeddings encode search history"
**Don't:** "Cumulative Embeddings"

Keep titles concise (one line on the poster column) but informative. Frame contributions as additive ("Towards a geometric framework…") rather than gap-filling ("X lacks Y").

## Section heading style

Underlined `h2` headings separate topics within columns. Using `border-bottom` keeps them compact:

```css
h2 {
  border-bottom: 2px solid var(--accent);
  padding-bottom: 1.5mm;
  display: inline-block; /* underline fits text width */
}
h2.full { display: block; } /* underline spans full column */
```
