# Lung Disease Classification from Chest X-rays (MSc Dissertation, 2023)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23065563.svg)](https://doi.org/10.5281/zenodo.23065563)

Deep learning experiments on the **NIH ChestX-ray14** dataset, carried out for my MSc in Big Data Technologies at Glasgow Caledonian University (dissertation supervised by Dr. Sajid Nazir).

The notebook compares three ImageNet-pretrained convolutional neural networks — **ResNet-50** (PyTorch), **MobileNet** and **VGG16** (TensorFlow/Keras) — for detecting lung disease in chest radiographs.

> **Status:** this repository preserves the original 2023 dissertation code as submitted. A corrected and extended version (see *Limitations and next steps*) is in progress.

## Dataset

- **Source:** [NIH Chest X-rays on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) (Wang et al., 2017)
- 112,120 frontal chest X-rays from 30,805 patients, labelled with up to 14 thoracic pathologies or "No Finding"
- The images (~42 GB) are **not included** in this repository; the notebook downloads them via the Kaggle API.

## Experiments

| Model | Framework | Task set-up | Data used | Train acc. | Val. acc. |
|---|---|---|---|---|---|
| ResNet-50 (fine-tuned) | PyTorch | Binary: any finding vs. "No Finding" | 11,000-image random sample (10,000 / 1,000 split) | 95.4% | 64.1% |
| MobileNet (frozen base) | Keras | Each label combination treated as a class (836 classes) | Full dataset (80/20 split) | 52.9% | 58.1% |
| VGG16 (frozen base) | Keras | Each label combination treated as a class (836 classes) | Full dataset (80/20 split) | 52.8% | 58.1% |

Training curves for each model are plotted inside the notebook.

## Limitations and next steps

Looking back at this work, several issues limit how far the numbers above can be trusted. I list them here openly because fixing them is the basis of the follow-up project.

1. **Wrong problem framing for MobileNet/VGG16.** ChestX-ray14 is a *multi-label* dataset (one image can show several diseases). Treating every label combination as its own class produced 836 classes and a softmax output; the ~58% validation accuracy is close to what a model gets by predicting "No Finding" for every image. The correct set-up is 14 sigmoid outputs with binary cross-entropy.
2. **Accuracy is the wrong metric.** With imbalanced classes, accuracy hides failure on rare diseases. The standard for this dataset is **per-class AUROC**.
3. **Overfitting in ResNet-50.** Training accuracy reached 95% while validation accuracy stayed near 64% and validation loss rose after the early epochs. Early stopping and stronger regularisation are needed.
4. **Patient leakage.** Splits were made by image, not by patient, so X-rays of the same person can appear in both training and validation. The dataset's official patient-wise split (`train_val_list.txt` / `test_list.txt`) should be used.
5. **Models are not directly comparable**, since ResNet-50 was trained on a different task and data subset from MobileNet and VGG16.

**Planned v2:** a single framework, multi-label training of ResNet, DenseNet, EfficientNet and MobileNet on the official split, per-class AUROC, class-imbalance handling, and Grad-CAM visualisations.

## How to run

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
