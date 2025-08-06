1. "Optimization as Interface" — Interactive Constraint Solver
Build a simple GUI or command-line tool where users type their output goals (e.g., “make it hotter but reduce emissions”), and the backend uses your neural constraint optimization model to update the inputs.
Backend: Your constraint optimization code
Frontend: Streamlit / CLI mock
Small dataset or even synthetic
Show a few demo cases + user-style experiments
✅ New angle: ML + user goals
✅ Fast to execute
✅ Publishable as HCI / AI tooling / optimization

---

3. Physics-Aware Diffusion Denoising
Use a physics equation (e.g., heat diffusion, wave propagation) as a “truth generator” and train a small denoiser model to recover clean state from noise-corrupted simulations.
Simulate dirty PDE solutions (heat, wave, fluid)
Train a small CNN or U-Net to denoise
Compare to traditional numerical smoothing
Bonus: Add constraints or boundary learning
✅ Fast, visually compelling
✅ Could be in applied physics + ML + PDE venues


# Physics-Aware Denoising with Diffusion from Scratch

This project uses a synthetic dataset generated from known PDEs (partial differential equations) and applies a deep learning model to recover clean physical fields from noisy versions. It builds on a custom **diffusion model from scratch**, without requiring external pretrained models or datasets.

## 🔬 Project Summary
We simulate noisy versions of physical phenomena governed by PDEs (e.g., the heat equation), and train a neural network (e.g., CNN or U-Net) to reconstruct the original, clean solution.

## ✅ Key Highlights
- **All data is synthetic**: no external datasets used
- **Physics-informed setup**: data generated from known PDEs (e.g., heat diffusion)
- **Uses stable diffusion from scratch**: custom-built, no pretrained models
- **No retraining required**: pretrained scratch diffusion is adapted directly to the denoising task

## 🧪 Workflow
1. **Simulate Clean Data**
   - Use 2D finite-difference heat equation solver
   - Store snapshots of clean field `U_clean`

2. **Corrupt with Noise**
   - Add Gaussian noise or salt-and-pepper to get `U_noisy`

3. **Train Neural Denoiser**
   - Model: CNN or U-Net
   - Input: `U_noisy`
   - Output: `U_clean`

4. **Compare to Baselines**
   - Traditional smoothing: Gaussian blur, median filter
   - Metrics: MSE, PSNR, SSIM, visual difference

5. **Optional Enhancements**
   - Add boundary condition masks
   - Use time-aware inputs
   - Try multiple PDEs (e.g., wave, Poisson)

## 📁 Dataset
- Shape: `64x64` 2D grids
- Format: `.npy` or `.png`
- Fields: `(U_noisy, U_clean)` pairs
- Dataset size: 1000 samples (expandable)

## 🧠 Model
- **Architecture**: 3–5 layer CNN or small U-Net
- **Loss Function**: MSE between output and `U_clean`

## 📊 Evaluation
| Method           | MSE ↓ | SSIM ↑ | PSNR ↑ |
|------------------|-------|--------|--------|
| Gaussian blur    |       |        |        |
| Median filter    |       |        |        |
| CNN/U-Net Model  |       |        |        |

Visuals include:
- Input vs. Ground Truth vs. Model Output
- Error maps (|prediction - truth|)

## 📚 Paper Outline
1. Abstract
2. Introduction
3. Related Work (denoising, PDE learning)
4. Method: simulation + training
5. Experiments + results
6. Discussion + conclusion

## 📦 Requirements
- Python 3.10+
- PyTorch
- NumPy, Matplotlib

---

This framework provides a lightweight but powerful bridge between PDE-based physics simulations and deep learning-based denoising. Perfect for visually compelling ML + physics experimentation.


---

5. Multi-Agent Sim to Learn Communication
Toy 2D agents try to pass a ball, but must learn to communicate a plan through a latent vector.
Use PyBullet or a 2D engine (even simple matplotlib sim)
Each agent gets partial info (e.g., obstacle location)
Must encode + decode a message to cooperate
Train with reward on success + message efficiency
✅ Publishable as an emergent communication or RL paper
✅ Code is small but the idea is hot

---


7. AI-Powered Calibration of Analog Sensors
Show that an NN can learn to calibrate noisy analog sensors (e.g., thermocouples, light sensors) better than a polynomial or linear fit.
Simulate realistic sensor drift or nonlinearity
Train MLP or 1D CNN to calibrate
Benchmark vs. traditional fitting
✅ Very practical
✅ Could submit to applied AI, embedded ML venues
✅ Can even run on MCU/Orin Nano later


---

9. Automatic Signature Similarity via Vision Embeddings
Turn your art authentication idea into a technical paper.
Use CLIP or ResNet to embed signatures
Compute similarity scores between known and unknown samples
Small dataset (10–100 images) is enough
Visualize t-SNE or PCA embeddings + similarity matrix
✅ Fast, novel use case
✅ Publishable in applied vision or digital humanities/forensics



# Brushstroke Similarity Scoring via Vision Embeddings

## Overview

This project explores a novel method for quantifying stylistic similarity between paintings by analyzing brushstroke texture using self-supervised vision embeddings. Rather than focusing on content, composition, or signature, we train a model to learn the *style identity* of painters, particularly the brushstroke characteristics that are hard to forge.

By using contrastive learning with models like CLIP or DINOv2, we embed full or cropped images of paintings and train the system to recognize stylistic similarity between works by the same artist (e.g., Louis Ritman) and distinguish them from others.

---

## Goal

Train a vision-based model to:

* Identify stylistic consistency across paintings by the same artist
* Compute a **brushstroke similarity score** between unknown and known works
* Enable scalable, reusable stylistic analysis for art authentication

---

## Methodology

### 🔍 Embedding Backbone

* Use frozen CLIP or DINOv2 models to extract 512–1024D embeddings from image crops
* These embeddings capture texture, edge, contrast, and brushstroke implicitly

### 🧪 Contrastive Learning (Triplet or InfoNCE)

* Train a small model to minimize distance between paintings by the same artist (positive pairs) and maximize distance from other artists (negative pairs)

### 📥 Data Setup

* \~10 high-resolution paintings per artist
* Crop each painting into 3–5 regions with visible brush texture
* Artists: Louis Ritman, José Velásquez, Jean Salabet, Michel Delacroix, etc.
* Total: \~300–500 image crops

### 🧠 Training Objective

Use contrastive loss:

* Anchor: A crop from one Ritman painting
* Positive: A crop from a different Ritman painting
* Negative: A crop from another artist

Result: The model learns to embed brushstroke style in latent space

---

## Metrics & Evaluation

| Metric                     | Description                                       |
| -------------------------- | ------------------------------------------------- |
| Cosine Similarity          | Between unknown painting and known artist cluster |
| Clustering (t-SNE / PCA)   | Show embedding space groups per artist            |
| ROC-AUC                    | Evaluate thresholding ability for artist match    |
| Human Alignment (Optional) | Compare scores to expert similarity judgments     |

---

## Applications

* **Art Authentication**: Detect stylistic consistency or outliers
* **Forgery Detection**: Flag suspicious paintings based on brushstroke deviation
* **Digital Humanities**: Stylometric analysis of under-studied artists
* **Collector Tools**: Provide score-based confidence metrics

---

## Bonus Extensions

* Train a second-layer MLP to output a \[0,1] similarity score
* Apply model to new or unlabeled paintings for exploration
* Add Grad-CAM or patch attention to visualize which regions drive style similarity

---

## Tools

* Python, PyTorch
* CLIP (OpenAI), DINOv2 (Meta)
* Matplotlib, Seaborn for visualization

---

## Status

✅ Idea finalized
✅ Model design defined (Option 2: contrastive learning)
🔜 Collect image crops & begin embedding + training phase

---

## Lead Author

Ricardo Calix, 2025

This project blends vision science, art authentication, and self-supervised learning for a practical, novel use case.

---

## Contact

If you’d like to collaborate or share datasets, reach out at [rcalix@rcalix.com](mailto:rcalix@rcalix.com)



---

11. Procedural Data Generation for Model Robustness
Show that procedurally generated training data (e.g., basic 2D or 3D shapes) can pre-train a model that adapts better to real-world data.
Simulate simple shapes with variable lighting/background
Train a ResNet on this
Fine-tune on small real dataset (e.g., CIFAR subset)
Show robustness gains
✅ Ties into game/graphics work if desired
✅ Can use Unity or Python simulation
✅ Publishable in data-centric AI venues

---



Paper Idea: “Neural Constraint-Based Password Auditor: A New Approach to Password Quality Enforcement”
✅ Core Idea:
Use a neural constraint optimization model to reverse-engineer plausible user passwords that satisfy common password rules — length, character mix, entropy — and show how bad policies can be gamed and how better constraints reduce guessability.
🧠 Key Insight:
Traditional password policies (e.g., “one uppercase, one number”) often produce predictable formats (e.g., Password1!)
Your model treats the rules as constraints and generates passwords that meet them
You measure how similar they are to real user-chosen passwords (e.g., leaked corp datasets or synthetically modeled ones)
You show that policies which encourage low-entropy “rule-compliance” lead to more predictable passwords
🧪 Method Steps:
Constraint-based Input Optimization Model
Inputs: latent seed vector
Constraints: must match a policy (e.g., regex, entropy floor, char set rules)
Output: synthetic password string that satisfies the constraints
Similarity Evaluation
Compare generated passwords to real-world datasets (e.g., RockYou, LinkedIn dumps)
Use Levenshtein, character entropy, n-gram distribution, etc.
Policy Weakness Audit
Run your model with several common policies
Show which ones produce most “guessable” passwords
Recommend better constraint design
📊 Experimental Setup:
Use synthetic data OR public leaks (RockYou, etc.)
Generate 10,000 passwords per policy
Score each batch using:
Entropy metrics
Markov likelihood based on real password models
Similarity to known leaks
🧩 Publishable Contributions:
First use of neural constraint optimization to generate passwords from policy rules
Novel method to audit password policies from the inside-out
Clear applied relevance + security implications
Tiny dataset needed, easy to simulate everything
🕒 Timeline:
Task	Time
Basic model reuse from your constraint code	1–2 days
Write 3–5 policies (regex + entropy)	1 day
Generate samples + metrics	2–3 days
Visualize results, write paper	3–4 days
📄 Possible Title Variants:
“How Secure Is Your Password Policy? A Constraint-Based Auditor Reveals the Answer”
“Neural Password Generator as Policy Critic: Constraint Optimization for Security”
“Learning to Game the Rules: Password Guessability under Policy Constraints”
