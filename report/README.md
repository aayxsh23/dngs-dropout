# Report (IEEE conference format)

- `main.tex`: the paper (IEEEtran, `conference`). `refs.bib`: references. `figures/`: PDF figures copied from `results/figures/` and `results/v2/figures/`.
- `NUMBERS.md`: every number in `main.tex`, with the committed file and line it comes from. `COMPLIANCE.md`: course requirements and where each is met.

Compile on Overleaf (upload this folder; compiler pdfLaTeX) or locally:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
