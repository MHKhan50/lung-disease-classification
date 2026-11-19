# Paper

**Class Weights Miscalibrate Chest X-ray Classifiers by a Predictable Amount: Calibration, Prevalence Shift and Subgroup Noise in a Reproducible ChestX-ray14 Baseline**

| File | What it is |
|---|---|
| `main.pdf` | The paper, also on Zenodo: [doi:10.5281/zenodo.23840865](https://doi.org/10.5281/zenodo.23840865) |
| `main.tex`, `references.bib`, `main.bbl`, `figures/` | LaTeX source (arXiv-ready: upload these files) |
| `analysis.py` | Extra analyses used in the paper; writes `results.json` and the figures |
| `seed42_check.py` | Patient-level interval and calibration for DenseNet121 seed 42; writes `seed42.json` |
| `results.json` | Every number in the paper that is not already in `v2.1/analysis/` |
| `train_label_counts.json` | Positive counts in the training split, used to recompute the class weights |

Reproduce, from the repository root (about 10 minutes on a laptop, mostly the patient-level bootstrap):

```bash
pip install -r v2.1/requirements.txt
python3 paper/analysis.py
python3 paper/seed42_check.py
cd paper && latexmk -pdf main.tex
```
