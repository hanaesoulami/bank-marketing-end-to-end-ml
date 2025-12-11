# 🏦 Bank Marketing Machine Learning Project

This project explores how we can use machine learning to help a bank predict whether a client will subscribe to a term deposit after a marketing campaign.  
I used this dataset to build a complete **end-to-end ML workflow**, from **data exploration and preprocessing** to **training and comparing different models**.

The dataset is real it comes from a Portuguese bank’s marketing campaigns, and it’s widely used for classification tasks in machine learning.

---

## 🎯 Project Goal

The main objective of this project was to:
> **Predict whether a client will subscribe to a term deposit based on personal, financial, and campaign-related information.**

I wanted to approach this like a real-world ML problem:
1. Start with raw, messy data.
2. Explore and visualize patterns.
3. Clean and transform it for modeling.
4. Try several models to see what works best.
5. Evaluate, compare, and explain the results.

---

## 📁 Project Structure

This project is divided into two main Jupyter notebooks:

1. **`01_EDA_and_Preprocessing.ipynb`** – Data exploration, visualization, cleaning, encoding, and scaling.
2. **`02_Modeling.ipynb`** – Model training, performance evaluation, and feature importance analysis.

---

## 📊 The Dataset

- **Source:** [UCI Machine Learning Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Records:** 11,162 clients  
- **Target:** `deposit` (1 = yes, 0 = no)  
- **Features:** 16 explanatory variables including:
  - `age`, `job`, `marital`, `education`, `balance`
  - `housing`, `loan`, `contact`, `month`, `day`
  - `duration`, `campaign`, `pdays`, `previous`, `poutcome`

Each record represents one client and their response to a marketing call from the bank.

---

## 🧭 Step 1: Exploratory Data Analysis (EDA)

I started by understanding the data its shape, types, and relationships between variables.  
Here’s what I explored:

- Checked for **missing values** → none found (clean dataset)
- Looked at **data types** → 7 numeric, 10 categorical
- Used **visuals** to understand patterns:
  - Violin plots (to see distributions of numeric variables)
  - Count plots (to check class balance)
  - Bar charts (to understand category frequencies)
  - Histograms (to look at feature distributions)

### 🧠 Insights from EDA
- **Age:** The majority of clients are between 30 and 50 years old.
- **Balance:** Most clients have small account balances, but a few have very high ones.
- **Duration:** The longer the last call, the more likely the client said “yes”.
- **Poutcome:** Clients with a previous campaign success are much more likely to subscribe again.
- **Month:** Most calls happen in May and August.
- **Contact type:** Most clients were contacted by **cellular**, which tends to have better results than telephone.

These patterns already gave me an idea of which variables might matter most later.

---

## 🧹 Step 2: Data Cleaning & Preprocessing

After the EDA, I cleaned and prepared the dataset for modeling.  
Here’s everything I did in this step:

### 🧾 Encoding
- Converted **binary columns** (`yes`/`no`) into **1 and 0**.
- Applied **one-hot encoding** to categorical features like:
  - `job`, `education`, `marital`, `month`, `contact`, `poutcome`
- Removed the original categorical columns after encoding.
- Dropped one dummy variable per category to avoid multicollinearity.

### 🧮 Feature Scaling
- Standardized all numeric columns using **StandardScaler** to give each feature the same scale and avoid bias in distance-based algorithms.

### 💾 Final Dataset
- Ended up with **43 numerical features**
- All categorical variables were converted to numeric
- Saved the clean version as `PreprocessedBank.csv`

At this stage, the data was ready for modeling.

---

## ⚙️ Step 3: Machine Learning Modeling

Once the data was cleaned, I moved on to training and comparing different machine learning models.

I wanted to test a variety of algorithms from simple linear ones to more complex ensemble models and see how each performed on this classification task.

### 🧠 Models I Trained
1. **K-Nearest Neighbors (KNN)**
2. **Naive Bayes**
3. **Logistic Regression**
4. **Polynomial Logistic Regression** (degrees 2 and 3)
5. **Support Vector Machine (SVM)**
6. **Decision Tree**
7. **Random Forest**
8. **XGBoost**

### 🧪 Training & Evaluation

- Split the data into **training (70%)** and **testing (30%)**
- Evaluated each model with:
  - **Accuracy**
  - **Recall**
  - **AUC (Area Under the ROC Curve)**
  - **Confusion Matrix**
  - **ROC Curve Visualization**

---

## 🧩 Step 4: Results

| Model | Accuracy | Recall | AUC |
|--------|-----------|--------|------|
| KNN | 0.74 | 0.68 | 0.83 |
| Naive Bayes | 0.70 | 0.53 | 0.79 |
| Logistic Regression | 0.81 | 0.79 | 0.89 |
| Polynomial LR (deg=2) | **0.84** | **0.83** | 0.91 |
| Polynomial LR (deg=3) | 0.83 | 0.80 | 0.90 |
| SVM | 0.82 | 0.89 | 0.89 |
| Decision Tree | 0.81 | 0.82 | 0.87 |
| Random Forest | 0.85 | 0.88 | 0.92 |
| **XGBoost** | **0.85** | **0.88** | **0.92** ✅ |

### 🏆 Best Model: **XGBoost**
XGBoost achieved the best overall results, balancing high recall and strong AUC performance.  
It was followed closely by Random Forest and Polynomial Logistic Regression.

---

## 🔍 Step 5: Feature Importance

To better understand *why* the model made its predictions, I analyzed feature importance using the XGBoost model.

### Top 10 Most Important Features:
1. **Duration** – Length of the last call (strongest predictor)
2. **Previous campaign success**
3. **Contact type (cellular/telephone)**
4. **Housing loan status**
5. **Month of contact**
6. **Marital status**
7. **Balance**
8. **Number of previous contacts**
9. **Education level**
10. **Default status**

🧠 The takeaway:  
Longer call durations and a successful previous campaign significantly increase the chance of a client saying *yes* to a deposit.

---

## 📈 Step 6: Key Insights

Here’s what I learned from the analysis and modeling:

- **Human interactions matter** – the longer the call, the higher the chance of conversion.
- **Previous relationships count** – clients who said “yes” before are more likely to say “yes” again.
- **Communication type is critical** – contacting clients via **cellular** is far more effective than telephone.
- **Seasonality exists** – most successful contacts happened in **May** and **August**.

---

## 🧰 Tools & Libraries

- **Python 3**
- **Jupyter Notebook**
- **pandas**, **numpy**, **matplotlib**, **seaborn**
- **scikit-learn**
- **xgboost**

---

