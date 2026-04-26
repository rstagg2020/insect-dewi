# DeWi: Deep-Wide Insect Classification

This repository contains the training and evaluation code for the DeWi (Deep-Wide) model architecture on the 100k VT Insect Dataset (2014-2022). 

## Training History & Architectural Evolution

To solve a massive Divergent Optimization issue where the Metric Loss was stable but the Cross-Entropy (CE) Loss was mathematically exploding (Logit Norm Explosion), the model architecture and training loop evolved through the following phases:

### Epoch 1 – 18 (Initial Run & Divergence Setup)
- **Status:** Standard fine-tuning from IP102 pre-trained weights.
- **The Issue:** The Triplet Margin metric loss successfully formed dense intra-class clusters. However, the unconstrained `nn.Linear` classifier began to struggle with the geometry. To maintain confidence on these tiny angular margins, it started scaling its weights toward infinity ($||W|| \to \infty$), setting the stage for a logit magnitude explosion.

### Epoch 18 – 24 (The Catapult & Explosion)
- **Status:** Resumed from the Epoch 18 local optimum. 
- **Numerical Stabilization Attempt:** **Label Smoothing (0.1)** was applied to the `nn.CrossEntropyLoss` to remove the asymptote that encourages infinite confidence.
- **The Bug:** A bug in the checkpoint resume logic accidentally overwrote the differential learning rate, forcing the backbone and the head to train at a 1:1 ratio. The combination of the unconstrained linear head and the rapid backbone updates catapulted the weights out of their local minimum. Accuracy plateaued around 25%, while Cross-Entropy loss exploded to **300+**.

### Epoch 25 – 30 (The Architectural Fix)
- **Status:** Training paused and architectural geometries were explicitly aligned.
- **Logit Normalization:** Replaced `nn.Linear` with a custom `CosineClassifier` layer. This mathematically bounds the logits by explicitly $L_2$-normalizing both the feature embeddings and the weight vectors before taking the dot product, physically preventing any future CE explosion.
- **Learnable Scale ($s$):** The normalized cosine similarity is scaled by a parameter $s$, changed from a static scalar to a `nn.Parameter` initialized at `20.0`, allowing the network to dynamically learn the optimal temperature of the distribution.
- **Asynchrony Management:** The learning rate adjustment logic in `train_vt.py` was fixed to explicitly maintain a **10:1 differential ratio**. The new head receives `new_lr` to rapidly learn the new boundaries, while the pre-trained ResNet backbone receives `new_lr * 0.1` to prevent feature destruction.
- **Strict Loading Bypass:** Modified `utils/auto_load_resume.py` to use `strict=False`. This allowed the script to intentionally drop the corrupted/exploded weights of the old linear head when resuming. 
- **Result:** Accuracy initially dropped to ~12% (due to random initialization of the new head) but rapidly climbed to 17% by Epoch 30. Most importantly, the Cross-Entropy loss plummeted and permanently stabilized at **~5.1 - 5.5**.

### Epoch 30 – 100 (Current Run)
- **Status:** With the mathematics stabilized and the divergence structurally prevented, the `end_epoch` limit was raised from 30 to 100. This phase is dedicated to allowing the new angular hyperplanes to fully converge and break past the previous 25% architectural ceiling.
