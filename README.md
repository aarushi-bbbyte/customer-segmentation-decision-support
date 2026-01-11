# Customer Segmentation & Business Decision Support Tool

## Overview
This project is an end-to-end data science application that segments customers based on historical purchasing behavior and enables business teams to simulate the impact of targeted engagement strategies.

The system combines unsupervised machine learning with an interactive web interface to support data-driven and responsible business decision-making.

---

## Problem Statement
Businesses often struggle to identify high-value customers, understand disengaged segments, and evaluate retention strategies before deployment.

This project addresses that problem by:
- Grouping customers into meaningful segments using behavioral data
- Translating technical clusters into business-friendly personas
- Simulating “what-if” scenarios to estimate revenue impact

---

## Key Features
- RFM-based customer segmentation
- K-Means clustering with feature scaling
- Cluster stability validation using Adjusted Rand Index (ARI)
- Business persona mapping for interpretability
- What-if strategy simulation with diminishing returns
- Interactive web application built using Streamlit
- Clear separation between historical analysis and forward-looking simulation

---

## Dataset
- **Name**: Online Retail Dataset  
- **Source**: UCI Machine Learning Repository  
- **Description**: Real-world transactional data from a UK-based online retail store (2010–2011)

The dataset includes transaction timestamps, purchase quantities, prices, and customer identifiers and is aggregated at the customer level to compute RFM features.

> **Note:** The dataset is not included in this repository due to file size constraints.  
> It can be downloaded from the UCI Machine Learning Repository and placed in the `data/` directory.

---

## Technology Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit

---

## Project Architecture
├── app.py               # Streamlit frontend for user interaction and visualization
├── segmentation.py      # Backend-style module containing data processing,
├── requirements.txt     # Python dependencies required to run the application
└── data/
    └── README.md   
---

## How the Application Works
1. Historical transaction data is processed to compute RFM features
2. Customers are clustered using unsupervised learning
3. Clusters are validated for stability and mapped to business personas
4. Users interact with the app to simulate engagement strategies
5. The app displays projected revenue impact without retraining the model

---

## Responsible AI Considerations
- Clustering results are used for decision support, not automated decision-making
- All projections are indicative and require human judgment
- Personas are designed to aid strategy, not label individuals permanently

---

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
---

