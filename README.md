# EcoHeno 2.0 — Decision Support System for Sustainable Hay Production

EcoHeno 2.0 is a machine learning–based decision support system designed to predict hay production and simulate packing-day scenarios in agricultural production units.

The system supports operational decision-making by estimating production outcomes under different packing conditions, allowing organizations to anticipate capacity constraints, detect bottlenecks, and improve resource allocation.

---

## Overview

Agricultural production systems frequently operate under logistical and capacity constraints that directly affect productivity. EcoHeno 2.0 addresses this challenge by integrating predictive analytics with an operational dashboard, enabling data-driven planning in forage production environments.

Rather than relying solely on descriptive metrics, the platform provides predictive insight into how operational variables influence production performance.

---

## Key Features

- Machine learning–based production prediction  
- Packing-day simulation (1–6 days)  
- Operational loss estimation  
- Bottleneck detection  
- Executive decision-support dashboard  
- Deployment-ready architecture  

---

## Predictive Model

The predictive engine was developed using supervised machine learning techniques to estimate hay production based on critical operational variables, including:

- Cut-day production volume  
- Packing day  
- Operational sector  
- Month  

The trained model is exported as a `.pkl` artifact to ensure reproducibility and facilitate deployment without requiring retraining.

---

## Technology Stack

- Python  
- Streamlit  
- Scikit-learn  
- Pandas  
- NumPy  
- Plotly  

---

## Project Structure

ECOHENO/
│
├── app/ # Streamlit application
├── data/ # Source datasets
├── models/ # Trained ML artifacts (.pkl)
├── notebooks/ # Model development notebooks
├── requirements.txt
└── README.md

---

## Installation

Install dependencies:

pip install -r requirements.txt

---

## Running the Application

Launch the dashboard locally with:

streamlit run app/app.py


---

## Operational Insight

EcoHeno 2.0 goes beyond production estimation by revealing structural constraints within the packing process.  

Simulation results may indicate capacity saturation scenarios, highlighting when current operational resources — such as reliance on a single packing operator — could limit scalability.

This enables proactive decision-making regarding workforce allocation and capacity expansion.

---

## Reproducibility

The repository includes both the trained model and the feature structure used during training, ensuring consistent predictions across environments.

---

## Strategic Value

EcoHeno 2.0 contributes to the digital transformation of agricultural production by combining machine learning with operational analytics.

The system aligns production efficiency with sustainable resource management, supporting more resilient agro-production units.

---

## Important Note

The simulation results reflect operational capacity constraints observed in real production environments.

Specifically, packing operations currently rely on limited human resources, which may restrict throughput under high cut-volume scenarios.

Therefore, the system should be interpreted not only as a predictive tool but also as an early indicator of structural bottlenecks that may require operational scaling.

---

## Authors

**Viviana Racero López**  
Universidad Católica Luis Amigó – Colombia  
ORCID: https://orcid.org/0000-0001-7779-2585  

**Weimar Cortés Montiel**  
Pontificia Universidad Javeriana Cali – Colombia  
ORCID: https://orcid.org/0009-0003-5427-7443  

---

## Citation

If you use this system in academic research or industrial applications, please cite the associated publication (currently under review).

---

## Research Status

EcoHeno 2.0 is part of an applied research initiative focused on scalable digital solutions for sustainable agricultural production and intelligent decision-support systems.

---

## License

This project is released for academic and research purposes only.
For industrial or commercial use, please contact the authors.

