# Paper

**Class weights miscalibrate chest X-ray classifiers by a predictable amount: calibration, prevalence shift and subgroup noise in a reproducible ChestX-ray14 baseline**

| File | What it is |
|---|---|
| `main.pdf` | The paper, also on Zenodo: [doi:10.5281/zenodo.23840865](https://doi.org/10.5281/zenodo.23840865) |
| `main.tex`, `references.bib`, `main.bbl`, `figures/` | LaTeX source (arXiv-ready: upload these files) |
| `analysis.py` | Extra analyses used in the paper; writes `results.json` and the figures |
| `results.json` | Every number in the paper that is not already in `v2.1/analysis/` |
| `train_label_counts.json` | Positive counts in the training split, used to recompute the class weights |

Reproduce, from the repository root (about 10 minutes on a laptop, mostly the patient-level bootstrap):

```bash
pip install -r v2.1/requirements.txt
python3 paper/analysis.py
cd paper && latexmk -pdf main.tex
```
