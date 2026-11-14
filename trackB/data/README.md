# Track B data and local caches

| Folder | Purpose |
|---|---|
| `images/` | Resized WikiChurches images, grouped by train/validation/test and style |
| `wikichurches/` | Original labels, split files, church metadata, and image author/licence metadata |
| `new_images/` | Six separately downloaded church photographs used for the application demonstration |
| `logs/` | Dataset download and notebook training logs, plus the completed-download marker |
| `cache/torch_cache/` | Cached ImageNet backbone weights, reused by the notebook |
| `cache/hf_verification/` | Checkpoint copy used to verify the Hugging Face upload |
| `cache/cuda_wheels/` | Installation downloads and chunks retained from setting up GPU PyTorch |
| `cache/helper_pycache/` | Compiled caches of the temporary setup helpers |

The dataset is already cached locally. Preserve its split/class folder layout to reuse it without downloading the images again. `organization_manifest.json` records where files and folders were moved during the track reorganisation. Local paths printed in historical logs refer to the layout at the time of that run.

For public dataset links, image licences, and lighter alternatives, see [the model card](../models/README.md#download-the-dataset-when-needed). Prediction using the trained checkpoint does not require this data directory. Downloaded data, logs, and caches are excluded from Git; the documentation remains tracked.
