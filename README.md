# 📌 EECS 6320 Final Project — Fairness-Aware Loan Approval

This repository contains our EECS 6320 final project on **fairness-aware machine learning for loan approval prediction** using the 21st Mortgage Corporation dataset.

The goal of this project is to:
- Build a baseline model for loan approval prediction
- Analyze bias across sensitive attributes
- Apply and compare multiple debiasing techniques

---

## ⚙️ Pipeline Overview

The project is structured as a modular pipeline:

- **Preprocessing & Baseline Modeling**
  - Data cleaning and feature selection
  - Binary classification (`action_taken`: approved vs denied)
  - Baseline models (Random Forest, MLP)

- **Fairness Evaluation**
  - Demographic Parity (selection rate gap)
  - Equal Opportunity (TPR gap)
  - False Positive Rate (FPR gap)

- **Debiasing Techniques**
  - **Oversampling**
    - Rebalances underrepresented groups
  - **MMD Distribution Alignment**
    - Reduces distribution shift across sensitive groups
  - **Adversarial Debiasing**
    - Learns representations invariant to sensitive attributes
  - **Threshold Selection**
    - Adjusts decision boundaries for fairness-performance trade-offs

---

## 📊 Sensitive Attributes

Fairness is evaluated across:
- Gender (`applicant_sex`)
- Race (`applicant_race_1`)
- Age (`applicant_age`)

---

## 📁 Repository Structure

- `Joseph - Preprocessing + Training` → Data cleaning, baseline models  
- `Abhinav - Oversampling` → Oversampling-based debiasing  
- `Asad - MMD Distribution Alignment` → MMD-based methods  
- `Mahin - Adversarial Debiasing` → Adversarial training  
- `Mohsin - Threshold Selection` → Decision threshold tuning  
- `Dataset` → Source data  

---

## 🚀 Key Contributions

- End-to-end fairness pipeline  
- Comparative analysis of debiasing techniques  
- Real-world dataset evaluation  
- Modular and extensible design  

---
