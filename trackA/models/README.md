# Track A — saved Melbourne models

| File | Purpose |
|---|---|
| `melbourne_best_model.joblib` | Random-forest pipeline saved by the notebook's first supervised model comparison |
| `melbourne_house_gradient_boosting.joblib` | Tuned gradient-boosting pipeline saved later in the notebook |
| `melbourne_house_mlp.pt` | PyTorch checkpoint containing `model_state_dict`, ordered `feature_names`, and target `log10(Price)` |
| `melbourne_house_mlp_preprocessing.joblib` | Imputer, scaler, and ordered feature names used by the MLP |

Use the model definitions and feature engineering in [the notebook](../notebook/neural_networks_trackA.ipynb). To reload the MLP, reconstruct `HousePriceMLP` with the checkpoint's feature count, load its state dictionary, apply the saved imputer and scaler in the original feature order, and call `eval()` before prediction. Convert predicted log10 prices back to AUD with `10 ** prediction`.

Each supervised pipeline must receive the same features used to fit it; inspect the corresponding notebook section before inference. These files were moved without changing their contents. The church classifier and its class labels are in [Track B](../../trackB/models/README.md).
