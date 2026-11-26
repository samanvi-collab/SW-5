# SW-5 Satellite Collision Risk Predictor
### **Team DevX | Star Forge**
> **"Know when to act, and when to wait."**

---

## Overview
Satellite operators receive Conjunction Data Messages (CDMs) and must decide days ahead whether to perform a collision avoidance manoeuvre. Because most tracking warnings are harmless false alarms, dangerous conjunction events are exceptionally rare and noisy. **SW-5** is an intelligent, cost-aware decision framework built to maximize operator confidence—minimizing unnecessary fuel-burning manoeuvres while ensuring zero missed high-risk encounters.

---

## Key Features & Approach
* **Zero Data Leakage:** Restricts analysis strictly to CDMs received **2+ days before closest approach (TCA)** to ensure realistic operational foresight.
* **Advanced Feature Engineering:** Extracts temporal trends from event message histories, tracking latest risk, trajectory trends, miss distance shifts, covariance uncertainties, and total CDM volume.
* **Cost-Aware Decision Engine:** Optimizes operational costs by classifying events into three distinct actions: **Manoeuvre**, **Defer** (wait for next CDM update), or **No Manoeuvre**.
* **LightGBM Core:** Outperforms naive risk thresholds by leveraging a robust gradient-boosted decision tree architecture tailored for highly imbalanced space safety datasets.

---

## Performance Results (Test Events)

Evaluated on the official ESA Kelvins challenge dataset test split (1,792 unique event IDs):

| Method | Missed Dangerous Events | False Alarms | PR-AUC |
| :--- | :---: | :---: | :---: |
| Baseline (Latest risk $\ge 10^{-6}$) | 3 | 56 | 0.392 |
| Baseline (Latest risk $\ge 10^{-4}$) | 52 | 5 | 0.392 |
| **Our Model (Threshold 0.04)** | **3** | **55** | **0.816** |

---

## Links & Live Demos
* **Primary Dashboard (Streamlit — Trained LightGBM Model):** [Launch Streamlit App](https://vt42vb6nfjn763zzrzkdhc.streamlit.app/) *(Evaluates the 1,792-event test split with trained model predictions)*
* **Interactive CDM Explorer (Netlify — Rule-Based Preview):** [Launch Netlify Preview](https://public-link-b658e9.netlify.app) *(Interactive threshold simulator across the full 2,167-event ESA CDM stream)*
* **Video Walkthrough:** [Watch Demo on YouTube](https://youtu.be/DuBK6UE2I0U?si=QHIbYXgS-O9bwux-)

---

## How to Run Locally

1. **Clone the repository and set up a virtual environment:**
   ```bash
   git clone [https://github.com/samanvi-collab/SW-5.git](https://github.com/samanvi-collab/SW-5.git)
   cd SW-5
   python -m venv .venv
   # On Windows PowerShell:
   .venv\Scripts\Activate.ps1












   
