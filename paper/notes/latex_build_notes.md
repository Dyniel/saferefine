# LaTeX Build Notes

The main LaTeX draft is:

```bash
docs/submission/saferefine_media_draft.tex
```

The bibliography file is:

```bash
docs/submission/saferefine_references.bib
```

This cluster session does not currently expose `pdflatex`, `latexmk`, or
`chktex` on `PATH`, so the draft was not compiled to PDF here. On a machine with
a standard TeX installation, build from `docs/submission/`:

```bash
pdflatex saferefine_media_draft.tex
bibtex saferefine_media_draft
pdflatex saferefine_media_draft.tex
pdflatex saferefine_media_draft.tex
```

Expected external figure paths are relative to `docs/submission/`:

```text
../../results/submission_figures/fig_gain_harm_tradeoff.png
../../results/submission_figures/fig_key_bootstrap_effects.png
../../results/submission_figures/fig_strict_exact_fallback.png
```

Before final submission:

- replace placeholder author names and affiliations;
- confirm Elsevier's current preferred `elsarticle` options;
- verify all bibliography metadata in a reference manager;
- add acknowledgements, funding, and CRediT roles;
- move very wide supplementary tables to supplement if the PDF is visually too
  dense.
