# biomltool
# 🧠 Animal Extinction Predictor using Random Forest & Widgets

This Jupyter Notebook provides an interactive dashboard that allows users to select an animal species and a year, then predict:
- Whether the animal is endangered at that time,
- And project a **potential extinction year** based on model predictions and declining birth/death rates.

---

## 📁 Table of Contents

1. [Project Overview](#project-overview)  
2. [How It Works](#how-it-works)  
3. [Requirements](#requirements)  
4. [How to Run](#how-to-run)  
5. [Features](#features)  
6. [Example Use Case](#example-use-case)  
7. [Limitations & Future Work](#limitations--future-work)

---

## 🧾 Project Overview

The model simulates birth and death rates for various animal species from 2000 to 2025, and then predicts whether a species is endangered using a Random Forest classifier.

It uses `ipywidgets` for interactivity and includes:
- Live prediction of endangered status
- Future simulation to estimate extinction year (if any)
- Visual trend plots for population indices

---

## ⚙️ How It Works

1. **Data Simulation**: Generates synthetic birth/death rates and calculates a birth-death ratio.
2. **Model Training**: Uses `RandomForestClassifier` to learn patterns in the endangered status based on features.
3. **Interactive Inputs**: Dropdown widgets allow user to select a species and year.
4. **Prediction**: Classifier predicts if the species is endangered.
5. **Extinction Forecasting**: Projects up to 50 years into the future to estimate potential extinction year.
6. **Visualization**: Displays a line graph of population trends, with major global disruptions (like COVID-19) highlighted.

---

## 📦 Requirements

Make sure the following Python libraries are installed:

```bash
pip install pandas numpy matplotlib scikit-learn ipywidgets
```

Also, enable widgets for Jupyter:

```bash
jupyter nbextension enable --py widgetsnbextension
```

---

## ▶️ How to Run

1. Launch Jupyter Notebook:
2. Open the notebook file (e.g., `biomltool.ipynb`).
3. Run all cells (Shift + Enter).
4. Use the dropdowns to:
   - Select a species
   - Choose a year
5. Click the **Predict Status** button to see the result, extinction forecast, and plot.

---

## 🌟 Features

- ✅ Predict endangered status per year
- 🔮 Simulate future and forecast extinction year
- 📊 Visual trend graph for each animal's population
- 📉 Highlights effects of global crises (like 2008 Recession, COVID-19)
- 🧠 Machine learning-powered predictions

---

## 🐾 Example Use Case

- **Species**: Elephant 
- **Year**: 2012 
- → The model will predict if tigers were endangered in 2020 and whether they might go extinct in the next 50 years based on trends.
- ![Demo Output](https://github.com/user-attachments/assets/c5b6172d-0766-4a77-a84d-6281e0157b1d)

---

## ⚠️ Limitations & Future Work

- Uses synthetic data; not real-world accurate.
- Extinction logic based on simplified heuristics (e.g., birth/death ratio decline).
- Future enhancement could involve:
  - Real-world conservation datasets (e.g., IUCN Red List API)
  - Time-series modeling (e.g., ARIMA, LSTM)
  - Enhanced UI with dashboards (e.g., Voila or Streamlit)

---

**📝 Author**: Srinand  
**📅 Last Updated**: June 2025
