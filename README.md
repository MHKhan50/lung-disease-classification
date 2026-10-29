# Lung Disease Classification from Chest X-rays (MSc dissertation 2023, corrected V2 2026)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23065563.svg)](https://doi.org/10.5281/zenodo.23065563)

Deep learning experiments on the **NIH ChestX-ray14** dataset, carried out for my MSc in Big Data Technologies at Glasgow Caledonian University (dissertation supervised by Dr. Sajid Nazir).

The notebook compares three ImageNet-pretrained convolutional neural networks — **ResNet-50** (PyTorch), **MobileNet** and **VGG16** (TensorFlow/Keras) — for detecting lung disease in chest radiographs.

> **Status:** the original 2023 dissertation code is kept unchanged in `Lung_Disease_Classification.ipynb` (V1). A corrected V2, trained and evaluated in October 2026, is in [`v2/`](v2/) and summarised below.

## V2 (October 2026): multi-label, patient-wise, per-class AUROC

V2 fixes the three main problems listed under *Limitations* below: it treats the task as **14 labels** (sigmoid outputs, binary cross-entropy with class weights), uses the **official patient-wise split** (`train_val_list.txt` / `test_list.txt`, with a validation set split by patient), and reports **per-class AUROC** on the official test set.

Article: [I rebuilt my MSc chest X-ray model](https://masmatixinnovations.co.uk/insights/lung-xray-v2-calibration.html)

- Models: DenseNet121 and ResNet50, ImageNet-pretrained, same pipeline, 224x224, AdamW (lr 1e-4), batch 128, early stopping on validation mean AUROC.
- Data: 77,719 training, 8,805 validation and 25,596 test images; no patient appears in more than one split (checked in the code).
- Run: one run, seed 42, Kaggle Tesla T4. Training took 47 minutes (DenseNet121) and 42 minutes (ResNet50); both stopped after 7 epochs, with the best weights from epoch 5.

| Model | Test mean AUROC | 95% bootstrap interval | Best validation mean AUROC |
|---|---|---|---|
| DenseNet121 | **0.801** | 0.796 to 0.806 | 0.831 |
| ResNet50 | 0.795 | 0.791 to 0.799 | 0.827 |

![Per-class AUROC](v2/analysis/per_class_auroc.png)

Per-class test AUROC (DenseNet121): highest for emphysema (0.912), hernia (0.892) and cardiomegaly (0.879); lowest for infiltration (0.695), pneumonia (0.711) and consolidation (0.728). Full table: [`v2/results/per_class_auroc.csv`](v2/results/per_class_auroc.csv).

### Reliability analysis

`v2/reliability_analysis.py` re-reads the saved test predictions (no GPU needed) and writes [`v2/analysis/`](v2/analysis/).

- **DenseNet121 vs ResNet50.** Paired bootstrap over test images (1,000 resamples): difference +0.006, 95% interval 0.003 to 0.010. The bootstrap resamples images, not patients, so the true interval is probably wider.
- **Calibration.** The raw probabilities are far too high: the mean predicted probability is 2.5 to 10.9 times the share of test images that have each label (for consolidation, 0.49 against 0.07). Mean expected calibration error (ECE) is 0.265. The class weights used in training are the likely cause. Per-class Platt scaling, fitted on one half of the test images and applied to the other, brings mean ECE to 0.005 and mean Brier score from 0.175 to 0.058, with mean AUROC unchanged (0.801).
- **Referral of uncertain images.** Keeping the 50% of images with the lowest prediction entropy and referring the rest raises DenseNet121 mean AUROC from 0.801 to 0.816 (0.805 at 90% kept, 0.811 at 70%). This used the raw, uncalibrated probabilities.

![Predicted probability vs prevalence](v2/analysis/calibration_gap.png)

### V1 vs V2

| | V1 (2023) | V2 (2026) |
|---|---|---|
| Task | 836 label combinations (MobileNet, VGG16) or binary (ResNet-50) | 14 labels, multi-label |
| Split | random by image (patient leakage) | official, by patient |
| Metric | accuracy (58.1% validation) | per-class AUROC (0.801 mean, test) |
| Models comparable | no | yes, same data and pipeline |

The V1 and V2 numbers measure different things and cannot be compared directly. V2 is the first result from this project that can be trusted.

### My notes

- Infiltration (0.695), pneumonia (0.711) and consolidation (0.728) had the lowest AUROC. On an X-ray all three appear as white opacity where air should be, so they overlap visually. The labels were also mined from report text, which may be least reliable for these three. My guess is that the image alone is not enough and clinical information would help, but I have not tested that.
- For a clinician, wrong percentages can be more harmful than a lower AUROC. Before recalibration the average predicted probability for consolidation was 49% while only 7% of test images had it, so almost every image would raise an alarm. After Platt scaling the mean ECE fell from 0.265 to 0.005 with AUROC unchanged, so any demo or app has to show recalibrated values.
- V1 reported 58% accuracy on 836 label combinations with a random image split. V2 treats the task as 14 labels, splits by patient and reports per-class AUROC. The class weights I added in V2 helped the rare labels during training, but they are also what inflated the probabilities, so that choice needs a calibration step after it.

### Limitations of V2

- One run and one seed per model; differences of a few thousandths may change with another seed.
- 224x224 grayscale images and horizontal flips only; no higher resolution or stronger augmentation.
- The recalibration was fitted and checked on halves of the official test set, split by image, not by patient. A clean version fits it on the validation set.
- NIH labels were extracted from radiology reports by text mining and contain errors. All images come from one hospital, so results may not transfer to other sites.
- Not a medical device and not for diagnosis.

### Reproduce V2

1. Training: import `v2/lung-v2-chestxray14.ipynb` into Kaggle, add the `nih-chest-xrays/data` dataset, choose a GPU and turn Internet on, then *Save Version, Save & Run All*. Set `SMOKE = True` first for a ten-minute check.
2. Analysis (Mac, Python 3.9+): `pip install -r v2/requirements.txt`, then `cd v2 && python3 reliability_analysis.py --results results --out analysis` and `python3 -m unittest discover -s tests`.

Model weights are not stored in this repository.

## V1 (2023): dataset and experiments

### Dataset

- **Source:** [NIH Chest X-rays on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) (Wang et al., 2017)
- 112,120 frontal chest X-rays from 30,805 patients, labelled with up to 14 thoracic pathologies or "No Finding"
- The images (~42 GB) are **not included** in this repository; the notebook downloads them via the Kaggle API.

### Experiments

| Model | Framework | Task set-up | Data used | Train acc. | Val. acc. |
|---|---|---|---|---|---|
| ResNet-50 (fine-tuned) | PyTorch | Binary: any finding vs. "No Finding" | 11,000-image random sample (10,000 / 1,000 split) | 95.4% | 64.1% |
| MobileNet (frozen base) | Keras | Each label combination treated as a class (836 classes) | Full dataset (80/20 split) | 52.9% | 58.1% |
| VGG16 (frozen base) | Keras | Each label combination treated as a class (836 classes) | Full dataset (80/20 split) | 52.8% | 58.1% |

Training curves for each model are plotted inside the notebook.

## V1 limitations and next steps

Looking back at this work, several issues limit how far the numbers above can be trusted. I list them here openly because fixing them is the basis of the follow-up project.

1. **Wrong problem framing for MobileNet/VGG16.** ChestX-ray14 is a *multi-label* dataset (one image can show several diseases). Treating every label combination as its own class produced 836 classes and a softmax output; the ~58% validation accuracy is close to what a model gets by predicting "No Finding" for every image. The correct set-up is 14 sigmoid outputs with binary cross-entropy.
2. **Accuracy is the wrong metric.** With imbalanced classes, accuracy hides failure on rare diseases. The standard for this dataset is **per-class AUROC**.
3. **Overfitting in ResNet-50.** Training accuracy reached 95% while validation accuracy stayed near 64% and validation loss rose after the early epochs. Early stopping and stronger regularisation are needed.
4. **Patient leakage.** Splits were made by image, not by patient, so X-rays of the same person can appear in both training and validation. The dataset's official patient-wise split (`train_val_list.txt` / `test_list.txt`) should be used.
5. **Models are not directly comparable**, since ResNet-50 was trained on a different task and data subset from MobileNet and VGG16.

All five are addressed in V2 above (item 3 through early stopping). Still planned: EfficientNet and MobileNet in the same pipeline, Grad-CAM visualisations, sex and age subgroup results, and a chest X-ray foundation model as a comparison.

## How to run V1

The notebook was developed on **Google Colab with a GPU**, which is the recommended way to run it.

1. Open `Lung_Disease_Classification.ipynb` in Google Colab and select a GPU runtime.
2. Create a Kaggle API token (Kaggle → Settings → *Create New Token*) and upload the downloaded `kaggle.json` to the Colab session. **Never commit this file to GitHub.**
3. Run the cells in order. The dataset download and unzip take a while (~42 GB).

To run locally instead:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## References

- Wang, X. et al. (2017). *ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases.* CVPR.
- He, K. et al. (2016). *Deep residual learning for image recognition.* CVPR.
- Howard, A. G. et al. (2017). *MobileNets: Efficient convolutional neural networks for mobile vision applications.* arXiv:1704.04861.
- Simonyan, K. & Zisserman, A. (2015). *Very deep convolutional networks for large-scale image recognition.* ICLR.

## Author

**Muzammal Hussain** — Software Engineer, MSc Big Data Technologies (Glasgow Caledonian University)

- GitHub: [MHKhan50](https://github.com/MHKhan50)
  
- Kaggle: [muzammalhussain11](https://www.kaggle.com/muzammalhussain11)
  
## Citation

If you use this code, please cite:

> Hussain, M. (2026). *Lung Disease Classification from Chest X-rays — MSc dissertation code (2023)* (Version v1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23065564
