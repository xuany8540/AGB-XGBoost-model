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
AGB-XGBoost-model/
├── code/ # Python modeling & analysis scripts
│ ├── Lasso.py # Lasso feature selection
│ ├── XGboost.py # XGBoost modeling
│ ├── RF.py # Random Forest model
│ ├── MLR.py # Multiple Linear Regression
│ ├── VIF.py # Multicollinearity filtering
│ ├── lag effect.py # Lag effect analysis
│ ├── nonlinear analysis.py # SHAP interaction effects
│ ├── partial correlation.py # Partial correlation
│ └── Spatial heterogeneity.py # GWR / residual analysis
│
├── data/ # Input data (sample only)
│ ├── Sample_data.xlsx # Training dataset
│ ├── biomass/2009-2023.tif # AGB predictions by XGBoost
│ ├── Pre/2009-2023.tif # Annual precipitation (mm)
│ ├── Tem/2009-2023.tif # Annual temperature (°C)
│ └── Lag effect analysis/
│ ├── Pre.tif
│ ├── Tem.tif
│ └── 生物量.tif # Target AGB (unchanged name)
│
├── results/ # Model output and figures
│ ├── Feature selection/
│ ├── RF/
│ ├── MLR/
│ ├── XGBoost/
│ ├── partial correlation analysis/
│ ├── shap_plots/
│ ├── Spatial heterogeneity/
│ └── Lag effect analysis/
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── .gitattributes # Git settings

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
# 📥 Input Sources

Remote Sensing: Landsat, MODIS (NDVI, LAI, FPAR, etc.)

Climate: Annual temperature and precipitation

Topography: DEM-derived slope, aspect, curvature

Human: Population, accessibility (if applicable)

Inventory: Forest resource survey polygons (dominant species, origin types)

Note: The forest resource inventory data were provided by the Zhangye Administration and Protection Center of Qilian Mountain National Park.
These data are classified and restricted by law and policy, thus cannot be publicly shared.
To enhance reproducibility, we provide a sample dataset (data/Sample_data.xlsx) in which the "biomass" column has been replaced with values ranging from 1–100, demonstrating that the code can run as intended without access to the confidential inventory data.

---
## 💾 Output Overview

- Annual AGB maps (2009–2023)

- Evaluation metrics: R², RMSE, MAE

- SHAP importance & interaction plots

- Residual distribution maps

- Lag effect curves and tables

---
## 📦 Data Availability

Due to GitHub's file size limit (100 MB per file), the full-resolution datasets generated in this project—including annual forest aboveground biomass (AGB) predictions (2009–2023) and remote sensing-derived climate variables (temperature and precipitation)—are not hosted in this repository.

Instead, the complete GeoTIFF dataset has been deposited on the open-access platform Zenodo for public download and citation:

🔗 Zenodo DOI:
10.5281/zenodo.16996502
---

## 📁 Dataset Contents

A total of four compressed archives are provided:

- biomass.zip: 15 annual GeoTIFF files (2009.tif to 2023.tif) of AGB predictions

- Pre.zip: Annual precipitation maps (unit: mm)

- Tem.zip: Annual temperature maps (unit: °C)

- Lag_effect_data.zip: Climate–AGB data for lag-effect analysis

General Properties:

  - Spatial resolution: 30 meters

  - Coordinate system: WGS 84 / UTM Zone 49N (EPSG:32649)

Units:

  - AGB: Mg/ha (megagrams per hectare)

  - Temperature: degrees Celsius (°C)

  - Precipitation: millimeters (mm)
  
 ---
## 📥 Download & Usage Instructions

Download and extract all .zip files to your local directory

Update the corresponding relative paths in the scripts to match your local file locations

To test the code logic, you may use the provided sample file data/Sample_data.xlsx
→ Run `Lasso.py` and `VIF.py` to replicate the variable selection workflow

---
 
## 🧠 Reproducibility Tips

Ensure CRS consistency (e.g., EPSG:32649)

Remove NaN or missing values before training

Run VIF.py and Lasso.py for optimal feature selection

Use SHAP analysis to interpret model predictions

 ---

##  🔗 Access

📦 GitHub Repository: AGB-XGBoost-model

📁 Sample data and code are fully included

📊 All result files are reproducible from scripts

 ---

## 📚 Citation

If you use this project or dataset in your work, please cite:

Code repository

Yang, X. (2025). AGB-XGBoost-model: Forest Aboveground Biomass Estimation Using Remote Sensing and Explainable Machine Learning. GitHub. https://github.com/xuany8540/AGB-XGBoost-model

Dataset

Yang, X. (2025). Forest Aboveground Biomass in Tianzhu County (2009–2023) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.16996502




