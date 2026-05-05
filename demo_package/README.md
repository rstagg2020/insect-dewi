# Insect-Dewi Classification Demo

This folder provides a ready-to-use demo for insect classification using the models developed in this repository.

## Setup
1.  Ensure you have the required dependencies:
    ```bash
    pip install torch torchvision pillow wandb
    ```
2.  **Download Weights**: Model weights are stored as WandB Artifacts to keep the repository lightweight. Use the provided script to download them:
    ```bash
    python download_weights.py --entity your_wandb_username
    ```
    *Note: This script assumes you have logged artifacts named `focal-best-model`, `linear-best-model`, etc., to your project.*

### How to log your models as Artifacts
If you haven't logged your best models yet, run this snippet once to "enable" the download script:
```python
import wandb
run = wandb.init(project="dewi-insect-classification", job_type="upload")
artifact = wandb.Artifact('focal-best-model', type='model')
artifact.add_file('focal/focal_checkpoint/dewi_resnet50_vt/best_model.pth')
run.log_artifact(artifact)
run.finish()
```
(Repeat for `linear`, `foc_tran`, and `standard` with the appropriate names).

## Running the Demo
Once the weights are downloaded to the `weights/` folder, run the inference script:

```bash
# Basic usage
python inference.py --image path/to/insect.jpg --pipeline focal
```

### Arguments:
- `--image`: Path to the input image.
- `--pipeline`: Select from `focal`, `linear`, `foc_tran`, `standard`.

## Model Logic
The script automatically:
1.  Detects the correct architecture (Standard vs. Transformer).
2.  Loads the weights from the `weights/` directory.
3.  Preprocesses the image (384x384 resize, normalization).
4.  Displays the Top 5 most likely insect classes with confidence percentages.