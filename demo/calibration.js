// Per-class Platt parameters fitted on the NIH ChestX-ray14 validation split (DenseNet121, seed 2)
// and the share of NIH test images with each finding.
export const CALIBRATION = {
  "labels": [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
  ],
  "a": [
    0.956491,
    0.646954,
    0.873138,
    1.01714,
    0.794397,
    0.845892,
    0.600852,
    0.685285,
    0.754514,
    0.729006,
    0.589341,
    0.636695,
    0.625441,
    1.279206
  ],
  "b": [
    -1.95478,
    -3.075491,
    -2.069135,
    -1.697224,
    -2.894288,
    -2.679809,
    -3.599821,
    -2.918106,
    -3.106821,
    -3.541789,
    -3.520247,
    -3.571142,
    -3.341388,
    -3.065776
  ],
  "prevalence": [
    0.1281,
    0.0418,
    0.182,
    0.2388,
    0.0683,
    0.0634,
    0.0217,
    0.1041,
    0.0709,
    0.0361,
    0.0427,
    0.017,
    0.0447,
    0.0034
  ]
};
