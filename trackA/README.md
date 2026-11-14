# Track A — Melbourne house-price modelling

This track contains data cleaning, exploratory analysis, supervised regression comparisons, and a PyTorch multilayer perceptron for Melbourne house prices. The regression target is `log10(Price)`; predictions must be converted with `10 ** prediction` to return prices in AUD.

## Files

| Location | Contents |
|---|---|
| [notebook/neural_networks_trackA.ipynb](notebook/neural_networks_trackA.ipynb) | Complete notebook: cleaning, EDA, supervised models, tuning, and neural-network comparison |
| [data/](data/) | `melbourne_house_prices.csv` and `melbourne_house_prices_clean.csv` |
| [figures/](figures/) | Missing values, outlier treatment, distributions, geographical comparisons, correlations, and regression evaluation |
| [models/README.md](models/README.md) | Saved regression pipelines, MLP checkpoint, and preprocessing details |
| [requirements.txt](requirements.txt) | Track A dependencies |

## Run

From the repository root, activate the existing `.venv` and install `python -m pip install -r trackA/requirements.txt`. Open the notebook and select that environment as its kernel. Its setup cell resolves the Track A folder whether the working directory is the repository root, `trackA/`, or `trackA/notebook/`.

The raw and cleaned CSV files and saved models are already present locally. Running all notebook cells repeats cleaning and training and can overwrite generated CSVs, plots, and model files. Reading the saved notebook outputs does not require training. Training and evaluation use the feature engineering and preprocessing defined in the notebook; a saved model does not accept arbitrary raw columns without those steps.

The neural network has two hidden layers of 64 units, ReLU activations and dropout 0.15. It uses imputation and standardisation, Adam, validation-based early stopping, and a held-out test set. The notebook also compares classical regression methods. Refer to the executed notebook for recorded results and the distinction between cross-validation and held-out test metrics.

## Data and licence

The raw dataset is the Melbourne Housing Market dataset identified in the notebook's dataset card. Keep the original dataset attribution and confirm its source licence before redistribution. The cleaned CSV is a derived dataset produced by the notebook. The repository's [MIT code licence](../LICENSE) does not override the dataset's licence.
