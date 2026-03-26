# Eco-Urbanomics: Vehicle CO2 Emissions & Policy Optimization

**Author:** Khoi Nguyen   
**Focus:** Data Science applied to Economics, Public Policy, and Inclusive Green Growth

---

## 🌍 Project Overview

As urban centers transition toward inclusive green growth, environmental policy requires data-driven tools to enforce regulations and forecast emissions. This project applies machine learning and deep learning to automotive data to support urban environmental policy, shifting the focus from a standard regression exercise to a tailored, policy-relevant classification system. 

The core of this repository is a **PyTorch Dual-Head Multi-Layer Perceptron (MLP)** that simultaneously:
1.  **Predicts exact CO2 emissions (g/km)** for proportional tax and tariff modeling.
2.  **Classifies vehicles as "High Emitters" (≥ 278 g/km)** to support strict regulatory boundaries like Low-Emission Zones (LEZs).

This dual approach ensures the model learns deep, generalized representations of vehicle specifications, creating a robust tool for analyzing climate change adaptation strategies and drafting tangible policy implications for developing urban economies like Vietnam.

---

## 📂 Repository Structure

```text
eco-urbanomics-co2/
│
├── data/
│   ├── raw/                 # Original CO2_Emissions_Canada.csv (7,385 vehicles)
│   └── processed/           # Cleaned data with engineered features & scaling applied
│
├── models/
│   ├── baseline_rf_clf.pkl
│   ├── baseline_rf_reg.pkl
│   └── carbon_predictor_nn.pth
│
├── notebooks/
│   ├── 1_eda.ipynb                 # Exploratory Data Analysis, class balance, & visual diagnostics
│   ├── 2_feature_engineering.ipynb # Categorical parsing, frequency encoding, & domain-specific math
│   ├── 3_baseline.ipynb            # Random Forest baseline (Regression & Classification)
│   └── 4_deep_learning.ipynb       # PyTorch Dual-Head MLP architecture & threshold tuning
│
├── outputs/                 
│   ├── baseline_*.png              # RF feature importance and evaluation plots
│   ├── eda_*.png                   # Distribution and correlation matrices
│   ├── nn_*.png                    # Training curves, ROC-AUC, and threshold tuning 
│   └── nn_predictions_output.csv   # Final model predictions with Emission Risk Scores
│
├── README.md
└── requirements.txt
```

---

## 🛠️ How to Use This Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/cokevspeps/eco-urbanomics.git
   cd eco-urbanomics
   ```

2. **Create conda env and install packages**
   ```bash
   conda create --prefix .\.conda python=3.11 -y
   conda activate .\.conda
   ```
   ```bash
   conda install -r requirements.txt
   ```
