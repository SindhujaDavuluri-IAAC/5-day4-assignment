---
license: mit
library_name: pytorch
pipeline_tag: image-classification
tags:
  - pytorch
  - resnet
  - architecture
  - architectural-style
  - built-environment
  - iaac-maai
  - ai-methods-1
metrics:
  - accuracy
  - f1
---

# Church Style Classifier: the architectural style of a church from one photo

This model looks at an **exterior photo of a church** and predicts its **architectural style** among six: Romanesque, Gothic, Renaissance, Baroque, Classical (Neoclassical) and Modern, with a probability for each. It is meant to **pre-sort large photo archives** for heritage experts. It is a **ResNet18** (11.2 M parameters) pretrained on ImageNet, with its layers frozen and a new last layer trained on the **WikiChurches** dataset. It was built as a course assignment for AI Methods I (IAAC MaAI 2026/27).

- **GitHub repository (full pipeline):** https://github.com/iaac-maai/5-day4-assignment <!-- CHECK the final repository name -->
- **Team 5:** Soumyajyoti Datta, Sindhuja Davuluri, Breno Sader, Diego Alpízar

## Model at a glance

![Test images with the true and predicted style](https://huggingface.co/iaac-maai/day4-team5-church-style-classifier/resolve/main/test_predictions.png)

*Twelve test images, eight correct and four mistakes (red). The mistakes stay close to the truth: a Renaissance façade predicted Baroque, a Baroque church with a massive medieval-looking tower predicted Romanesque, and medieval churches swapped between Romanesque and Gothic. Low probabilities (31 %, 46 %) show that the model is often unsure even when it is right.*

| Model | Accuracy (test, 788 images) | Macro F1 (test) |
|---|---|---|
| Baseline: always the majority class | 0.190 | 0.053 |
| **This model** | **0.555** | **0.548** |

![Test metrics vs baseline](https://huggingface.co/iaac-maai/day4-team5-church-style-classifier/resolve/main/test_vs_baseline.png)

*On the 788 test images, the network is about three times better than always predicting the majority class, on every metric (accuracy 0.555 vs 0.190, macro F1 0.548 vs 0.053).*

## Intended use

- **Intended users and use:** heritage offices, architecture archives and researchers who want to **pre-sort** many exterior photos of churches by style, with an expert checking the results. Photos with a top probability below **60 %** should always go to an expert.
- **Out-of-scope uses:** buildings that are not churches; interiors, details, plans or drawings; dating or attributing a specific building; any heritage-protection, legal or funding decision without an expert; churches from regions that are rare in the training data (outside Europe) without checking the results first.

## Model details

| | |
|---|---|
| **Task** | Image classification (one style per image) |
| **Architecture** | ResNet18 with all pretrained layers frozen and a new last layer `nn.Linear(512, 6)` |
| **Parameters** | 11.2 M, of which 3,078 trainable (the last layer) |
| **Base model** | `torchvision` ResNet18, `ResNet18_Weights.DEFAULT`, pretrained on ImageNet-1k |
| **Input** | One RGB image, resized to 256 px, centre-cropped to 224 × 224 and normalised with the ImageNet mean and standard deviation (`ResNet18_Weights.DEFAULT.transforms()`) |
| **Output** | One probability per class (softmax over 6 logits), in the order below |
| **Files** | `model.pt` (weights, `state_dict`), `classes.json` (class order), `training_history.csv`, `experiments.csv`, figures, `figure_credits.csv` |
| **Library versions** | torch 2.14.1, torchvision 0.29.1, Python 3.14 |

### Classes

| ID | Class | Definition (what counts) |
|---|---|---|
| 0 | `baroque` | Baroque churches (17th–18th century, also Rococo): curved façades, rich ornament, domes, dramatic light |
| 1 | `classical` | WikiChurches' *Classical Architecture*: mainly Neoclassical churches, with temple fronts, columns, pediments and domes |
| 2 | `gothic` | Gothic churches (12th–16th century, incl. Brick Gothic and regional variants): pointed arches, buttresses, tall windows and towers |
| 3 | `modern` | WikiChurches' *Modern Architecture*: 20th- and 21st-century churches (incl. Expressionism, Modernism) |
| 4 | `renaissance` | Renaissance churches (15th–17th century): symmetry, classical orders, round arches, domes |
| 5 | `romanesque` | Romanesque churches (10th–12th century): thick walls, round arches, small windows, massive towers |

Labels come from the style of the church in **Wikidata**, regrouped by WikiChurches into these six first-level classes. Revival styles (e.g. Neo-Gothic) are labelled with their historical style.

## Training data

| | |
|---|---|
| **Dataset** | WikiChurches-6 (Barz and Denzler, *WikiChurches: A Fine-Grained Dataset of Architectural Styles with Real-World Challenges*, NeurIPS 2021 Datasets and Benchmarks) |
| **Source** | [Zenodo, DOI 10.5281/zenodo.5166987](https://doi.org/10.5281/zenodo.5166987) |
| **Licence** | Dataset CC BY-SA 4.0; every image keeps its Wikimedia Commons licence (CC BY-SA, CC BY, CC0, public domain…) |
| **Size** | Official splits, made **by church** (no church in two splits). Our reproducible subset (seed 42): **train 1,552** (at most 300 per class; Renaissance has only 64), **validation 306** (about 50 per class), **test 788** (at most 150 per class). |
| **Coverage** | Exterior photos of churches from Wikipedia/Wikidata, mostly in Germany and neighbouring European countries |

## Training procedure

- **Settings:** Adam, learning rate 0.001 for the new layer, batch 32, up to 20 epochs; **class-weighted** cross-entropy loss (weight 4.04 for Renaissance, 0.86–0.90 for the other classes, inversely proportional to the number of training images).
- **Augmentation and regularisation:** random resized crops (scale 0.7–1), horizontal flips, brightness and contrast jitter; frozen backbone with BatchNorm in evaluation mode; early stopping on the validation loss (patience 4).
- **Experiments (validation set):** a class-weighted loss for the rare styles, and fine-tuning `layer4` with a learning rate of 0.0001. Validation macro F1: frozen 0.575, **frozen + class weights 0.608 (selected)**, + fine-tuned `layer4` 0.570.
- **Hardware and time:** laptop NVIDIA RTX 4050 GPU, 5.5 minutes for the selected model (20 epochs).
- **Model selection:** the weights with the lowest validation loss of the selected experiment; the test set was used once, at the end.

![Training curves](https://huggingface.co/iaac-maai/day4-team5-church-style-classifier/resolve/main/training_curves.png)

*Both frozen models learn quickly and level off after about 10 epochs; the class-weighted model reaches the lowest validation loss (1.05). Fine-tuning `layer4` (epochs 21–26) overfits: the training accuracy jumps to 96 % while the validation loss rises, so we kept the frozen model with class weights.*

## Evaluation

| Metric (test, 788 images) | Baseline | This model |
|---|---|---|
| Accuracy | 0.190 | 0.555 |
| Balanced accuracy | 0.167 | 0.550 |
| Macro F1 | 0.053 | 0.548 |

**Per class:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| baroque | 0.512 | 0.433 | 0.469 |
| classical | 0.429 | 0.375 | 0.400 |
| gothic | 0.545 | 0.567 | 0.556 |
| modern | 0.841 | 0.835 | 0.838 |
| renaissance | 0.449 | 0.485 | 0.466 |
| romanesque | 0.523 | 0.607 | 0.562 |

![Confusion matrix](https://huggingface.co/iaac-maai/day4-team5-church-style-classifier/resolve/main/confusion_matrix.png)

**Where it fails:** Modern churches are recognised in 83 % of cases. The other mistakes stay inside two families: the medieval styles (36 Gothic churches predicted Romanesque, 26 Romanesque predicted Gothic) and the styles that share the classical vocabulary of columns, pediments and domes (Renaissance, Baroque and Classical, confused among themselves). Mistakes across the families are rare. Many of the most confident mistakes are churches that mix building phases, revival styles, regional features such as onion domes, or debatable labels; close-up views of a façade detail also fail.

![Validation mistakes](https://huggingface.co/iaac-maai/day4-team5-church-style-classifier/resolve/main/misclassified.png)

### On churches that are not in the dataset

![Six new churches](https://huggingface.co/iaac-maai/day4-team5-church-style-classifier/resolve/main/new_churches.png)

*On six churches that are not in the dataset, the model is right on three. The three mistakes stay inside the expected family; the 60 % threshold sends two of them to an expert, but La Mercè (Baroque, predicted Renaissance at 72 %) passes, so accepted photos should also be sampled.*

## Limitations and biases

- **Only church exteriors, mostly European.** The model has never seen other building types and few churches outside Europe.
- **Six broad classes.** Revival styles, regional variants and churches with several building phases are forced into one class; the labels come from Wikidata and were not checked by experts.
- **Photo conditions.** Training photos are mostly sunny, frontal, daytime views of whole buildings from Wikipedia; interiors, details, night, snow or drone views may fail.
- **Small and imbalanced rare classes.** Renaissance has only 64 training images; per-class scores are uncertain by several points.

## Ethics and privacy

The images show buildings; some contain passers-by, but they are small and incidental, and all images were already published on Wikimedia Commons under free licences. The model makes no prediction about people.

## How to use

```python
import torch
from torch import nn
from torchvision import models
from PIL import Image
from huggingface_hub import hf_hub_download

REPO = "iaac-maai/day4-team5-church-style-classifier"
CLASSES = ["baroque", "classical", "gothic", "modern", "renaissance", "romanesque"]   # output order

model = models.resnet18()                                     # the same architecture...
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))      # ...with the same last layer
model.load_state_dict(torch.load(hf_hub_download(REPO, "model.pt"), map_location="cpu"))
model.eval()

transform = models.ResNet18_Weights.DEFAULT.transforms()      # the same preprocessing as in training
image = Image.open("church.jpg").convert("RGB")
with torch.no_grad():
    probabilities = torch.softmax(model(transform(image).unsqueeze(0)), dim=1)[0]
print(dict(zip(CLASSES, [round(p, 3) for p in probabilities.tolist()])))
```

⚠️ `.pt` files are Python pickles and can execute code when loaded: only load models from sources you trust. Use the library versions listed above.

## Reproducibility

The full pipeline is in the GitHub repository: https://github.com/iaac-maai/5-day4-assignment <!-- CHECK -->. Run `notebook/neural_network.ipynb` with the packages in `requirements.txt` (seed 42 everywhere).

## Authors and licence

Soumyajyoti Datta, Sindhuja Davuluri, Breno Sader, Diego Alpízar. Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model released under the MIT License. The data is subject to its own licence: WikiChurches is CC BY-SA 4.0 (Barz and Denzler, 2021), and every image keeps its Wikimedia Commons licence; the authors and licences of the images shown in this card are listed in `figure_credits.csv`.
