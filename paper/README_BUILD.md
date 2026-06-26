# SafeRefine Full Submission Bundle

This directory is self-contained for manuscript editing.

## Main Files

- `main.tex`: main MedIA manuscript draft.
- `supplement.tex`: supplementary material with full tables.
- `saferefine_references.bib`: working BibTeX bibliography.
- `figures/`: main figures and qualitative case panels.
- `tables/`: CSV, Markdown, and LaTeX tables used by the manuscript and supplement.
- `notes/`: highlights, cover letter draft, reviewer defense, and checklist.

## Build

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

pdflatex supplement.tex
pdflatex supplement.tex
```

This cluster shell did not expose `pdflatex` or `latexmk`, so PDF compilation was not performed here.

## Final Manual Edits Before Upload

- Replace placeholder author names, affiliations, CRediT roles, funding, and acknowledgements.
- Verify every BibTeX entry in a reference manager.
- Check whether wide supplementary tables need landscape/sidewaystable layout.
- Confirm current Elsevier/MedIA formatting requirements before upload.
