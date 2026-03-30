# Causal and Active Learning-Based Counterfactual Chest X-ray Generation

> Supporting Clinical Decision-Making in Lung Disease through High-Fidelity Counterfactual Image Generation

This repository contains the research implementation of **Causal Generative Modelling** and **Active Learning** for pulmonary imaging datasets (e.g., MIMIC-CXR). The project aims to support clinical decision-making through high-fidelity counterfactual image generation.

This implementation extends the modular **Deep Structural Causal Model (DSCM)** framework for medical imaging, enabling high-fidelity counterfactual generation, combined with **Structural Variational Analysis (SVP)** for data efficiency and sample selection.

---

## 📦 Project Structure

```
.
├── causal/                      # Core causal generative model code
│   ├── pgm/                     # Probabilistic Graphical Models & SCM mechanisms
│   │   ├── dscm.py              # Deep Structural Causal Model module
│   │   ├── flow_pgm.py          # Flow mechanism implementation (based on Pyro)
│   │   ├── pgm_train.sh         # PGM model training script
│   │   ├── aux_train.sh         # AUX auxiliary model training script
│   │   ├── cf_train_test.sh     # Integrated Counterfactual (CF) training/testing
│   │   └── utils_pgm.py         # Graphical model utility classes
│   ├── vae.py                   # Hierarchical VAE (HVAE) definition
│   ├── trainer.py               # Training logic for imaging causal mechanisms
│   ├── datasets.py              # Dataset definitions (MIMIC, etc.)
│   ├── main.py                  # Main training entry point
│   ├── run_mimic.sh             # MIMIC-CXR HVAE experiment script (Baseline)
│   └── run.sh                   # General execution script
├── svp/                         # Structural Variational Analysis & Active Learning
│   ├── svp/mimic/               # MIMIC-CXR specific analysis package
│   │   ├── active.py            # Active learning/filtering logic
│   │   ├── coreset.py           # Coreset selection methods
│   │   ├── models.py            # SVP related model architectures
│   │   └── train.py             # SVP training script
│   ├── run_svp.sh               # SVP experiment execution script
│   └── setup.py                 # Environment setup script
└── README.md                    # Project documentation
```

---

## 🚀 Core Training Pipeline

The training follows a **strict phase-based pipeline**. Please follow these steps in order:

### 1. Step-wise Causal Model Training

#### Phase A: Pre-train HVAE Base Model (Baseline)

First, train the core image generation mechanism in the `causal` directory:

```bash
cd causal
bash run_mimic.sh your_experiment_name
```

#### Phase B: Train PGM and AUX Models

After HVAE training is complete, navigate to the `pgm` folder to train the probabilistic graphical mechanism and auxiliary models:

```bash
cd causal/pgm
bash pgm_train.sh   # Train PGM mechanism
bash aux_train.sh   # Train AUX auxiliary model
```

#### Phase C: Integrated CF Model Training

Once PGM and AUX are ready, perform the final integrated counterfactual training and evaluation:

```bash
cd causal/pgm
bash cf_train_test.sh
```

---

### 2. SVP Image Filtering Pipeline

After completing **Phase A** (HVAE Baseline), execute the following to refine your data:

**Step 1 — Generate Images:** Use the trained HVAE Baseline model to generate candidate counterfactual Chest X-rays.

**Step 2 — SVP Selection:** Run the SVP script to perform structural variational analysis and filter high-quality or representative samples:

```bash
cd svp
bash run_svp.sh
```

---

## 🛠️ Requirements

A Python 3.8+ virtual environment is highly recommended:

```bash
# Install base dependencies
pip install -r requirements.txt

# Install SVP toolkit in editable mode
cd svp
pip install -e .
```

---

## 📝 Citation

If you find this code useful for your research, please cite the following publication:

```bibtex
@InProceedings{pmlr-v202-de-sousa-ribeiro23a,
  title     = {High Fidelity Image Counterfactuals with Probabilistic Causal Models},
  author    = {De Sousa Ribeiro, Fabio and Xia, Tian and Monteiro, Miguel and Pawlowski, Nick and Glocker, Ben},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages     = {7390--7425},
  year      = {2023},
  volume    = {202},
  series    = {Proceedings of Machine Learning Research},
  url       = {https://proceedings.mlr.press/v202/de-sousa-ribeiro23a.html}
}
```
