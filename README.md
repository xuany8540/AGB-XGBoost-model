# 🌲 AGB-XGBoost-model: Forest Aboveground Biomass Modeling Project

This project develops a regional aboveground biomass (AGB) inversion model for forest ecosystems in Tianzhu County. By integrating multi-source remote sensing data and machine learning algorithms, it aims to produce high-accuracy and interpretable AGB estimates. The pipeline includes feature selection (Lasso), model training (XGBoost, RF, MLR), and post-analysis (SHAP, partial correlation, heterogeneity, lag effects).

---

## 📌 Project Objectives

This project uses remote sensing indices, terrain, climate, and anthropogenic indicators to:

- Identify key drivers affecting the spatial distribution of forest AGB;
- Compare multiple modeling techniques (XGBoost, RF, MLR);
- Generate annual AGB prediction maps (2009–2023);
- Quantify linear and nonlinear relationships;
- Explore spatial heterogeneity and climate lag effects;
- Support ecological management and carbon stock assessment.

---

## 🗂️ Directory Structure

```
AGB-XGBoost-model/
├── code/                           # Python modeling & analysis scripts
│   ├── Lasso.py                    # Lasso feature selection
│   ├── XGboost.py                  # XGBoost modeling
│   ├── RF.py                       # Random Forest model
│   ├── MLR.py                      # Multiple Linear Regression
│   ├── VIF.py                      # Multicollinearity filtering
│   ├── lag effect.py               # Lag effect analysis
│   ├── nonlinear analysis.py       # SHAP interaction effects
│   ├── partial correlation.py      # Partial correlation
│   └── Spatial heterogeneity.py    # GWR / residual analysis
│
├── data/                           # Input data (sample only)
│   ├── Sample_data.xlsx            # Training dataset
│   ├── biomass/2009-2023.tif       # AGB predictions by XGBoost
│   ├── Pre/2009-2023.tif           # Annual precipitation (mm)
│   ├── Tem/2009-2023.tif           # Annual temperature (°C)
│   └── Lag effect analysis/
│       ├── Pre.tif
│       ├── Tem.tif
│       └── 生物量.tif              # Target AGB (unchanged name)
│
├── results/                        # Model output and figures
│   ├── Feature selection/
│   ├── RF/
│   ├── MLR/
│   ├── XGBoost/
│   ├── partial correlation analysis/
│   ├── shap_plots/
│   ├── Spatial heterogeneity/
│   └── Lag effect analysis/
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitattributes                 # Git settings
```

---

## 🧭 Workflow Diagram

![Workflow](results/Technical%20route.png)

---

## ⚙️ Installation Instructions

### 1. Python Environment

- Python Version: **3.12 (64-bit)**
- Recommended: Use `venv` or `conda` for environment isolation

### 2. Install Dependencies

If `requirements.txt` is provided, install packages with:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, auto-generate one using:

```bash
pip install pipreqs
pipreqs ./AGB-XGBoost-model --encoding=utf-8 --force
```

---

## 🗃️ Data Description

### 📥 Input Sources

- **Remote Sensing**: Landsat, MODIS (NDVI, LAI, FPAR, etc.)
- **Climate**: Annual temperature and precipitation
- **Topography**: DEM-derived slope, aspect, curvature
- **Human**: Population, accessibility (if applicable)
- **Inventory**: Forest resource survey polygons (dominant species, origin types)

> **Note**: Due to data confidentiality, full inventory data is not shared. A sample dataset (`Sample_data.xlsx`) is provided for demonstration.

---

## 💾 Output Overview

- Annual AGB maps (2009–2023)
- Evaluation metrics: R², RMSE, MAE
- SHAP importance & interaction plots
- Residual distribution maps
- Lag effect curves and tables

---

## 🧠 Reproducibility Tips

- Ensure CRS consistency (e.g., EPSG:32649)
- Remove NaN or missing values before training
- Run `VIF.py` and `Lasso.py` for optimal feature selection
- Use `SHAP` analysis to interpret model predictions

---

## 🔗 Access

- 📦 GitHub Repository: [AGB-XGBoost-model](https://github.com/xuany8540/AGB-XGBoost-model)
- 📁 Sample data and code are fully included
- 📊 All result files are reproducible from scripts

---

## 📚 Citation

If you use this project in your work, please cite:

> Yang, X. (2025). AGB-XGBoost-model: Forest Aboveground Biomass Estimation Using Remote Sensing and Explainable Machine Learning. GitHub Repository. https://github.com/xuany8540/AGB-XGBoost-model

---

## 📮 Contact

For any inquiries or contributions, please open an issue on GitHub.
