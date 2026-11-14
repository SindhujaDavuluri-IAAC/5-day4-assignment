# Neural network assignments — Team 5

This repository contains two independent assignments for AI Methods I, IAAC MaAI 2026/27.

| Track | Task | Start here |
|---|---|---|
| **Track A — Melbourne** | Clean housing data, compare supervised regressors, and train a neural network to predict `log10(Price)` | [Track A README](trackA/README.md) · [Notebook](trackA/notebook/neural_networks_trackA.ipynb) |
| **Track B — Church classifier** | Classify church exterior photographs into six architectural styles using ResNet18 | [Track B README](trackB/README.md) · [Notebook](trackB/notebook/neural_network.ipynb) · [Model card](trackB/models/README.md) |

```text
trackA/
  README.md
  requirements.txt
  notebook/neural_networks_trackA.ipynb
  data/                         Melbourne raw and cleaned CSV files
  figures/                      Housing EDA and regression plots
  models/                       Regression pipelines, MLP, and preprocessing
trackB/
  README.md
  requirements.txt
  notebook/neural_network.ipynb
  data/                         Church images, metadata, logs, and local caches
  figures/                      Classifier data, training, and evaluation plots
  models/                       Classifier weights, model card, inference script, and results
  scripts/                      Hugging Face synchronisation helper
```

Use the existing root `.venv` or create one and select it as the notebook kernel. Install only the dependencies for the track you need:

```bash
python -m pip install -r trackA/requirements.txt
# Or, for the church classifier:
python -m pip install -r trackB/requirements.txt
```

The root `requirements.txt` includes both tracks. Each notebook resolves its own track folder when launched from the repository root, its track folder, or its notebook directory. Existing outputs and trained model files were preserved during organisation; neither model needs retraining because files moved.

The church classifier is published on [Hugging Face](https://huggingface.co/iaac-maai/day4_team5_church_style_classifier). From the repository root, predict using the local checkpoint without downloading the training images:

```bash
python trackB/models/predict.py "path/to/church.jpg" --model-dir trackB/models
```

Downloaded data, caches, and model weights are excluded from new Git additions by `.gitignore`. They remain available in the local track folders. Code uses the [MIT licence](LICENSE); datasets and photographs retain their own licences, described in the track documentation.
