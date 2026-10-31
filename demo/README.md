# Browser demo

Source of the live demo at [huggingface.co/spaces/MHkhan50/chest-xray-14-findings](https://huggingface.co/spaces/MHkhan50/chest-xray-14-findings).

DenseNet121 (V2.1, seed 2) runs in the browser with ONNX Runtime Web, so images are never uploaded. The app reproduces the training preprocessing (Pillow-style grayscale and bilinear resize to 224x224, ImageNet normalisation), applies the per-class Platt calibration fitted on the validation set, and draws class activation maps (equal to Grad-CAM for this architecture after normalisation).

| File | What it is |
|---|---|
| `index.html`, `app.js` | The app |
| `calibration.js` | Platt parameters from `v2.1/analysis/platt_params.json` and test-set prevalence |

The ONNX model and the three example images are produced by [`v2.1/chestx-ray14-v2-1-onnx-export.ipynb`](../v2.1/chestx-ray14-v2-1-onnx-export.ipynb) and stored in the Space, not here. In the browser the app's probabilities match the Python reference outputs within 0.2 percentage points on all three examples.

To run it locally, put `chest_xray_densenet121.onnx` and `example_1.png` to `example_3.png` in this folder and serve it with `python3 -m http.server`.

Research demo only. Not a medical device and not for diagnosis.
