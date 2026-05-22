# 💳 Credit Card Behavior Score (Bank A Case Study) - Convolve Round 1

This repository contains the codebase, analysis files, and documentation for the **Credit Card Behavior Score** project, developed for Convolve Round 1. 

## 📖 Business Context & Problem Statement

Bank A issues credit cards and deploys advanced ML frameworks to manage early risks and profitability. The Bank aims to build a robust risk management framework for its existing credit card customers (whose cards are open and not past due) to predict their probability of defaulting on future payments. This predictive model is known as the **"Behavior Score"**. 

In the context of this project, a **default** occurs when a credit card user fails to meet their payment obligations (e.g., minimum payment or full balance) for a prolonged duration, typically 90 or 120 days past due. This project aims to assess this risk and provide actionable insights.

## 📁 Dataset Details

The provided data contains a snapshot of the Credit Card portfolio of Bank A:
- **Development Data (`Dev_data_to_be_shared.csv`):** 96,806 historical credit card records along with a `bad_flag` target (1 = default, 0 = non-default).
- **Validation Data (`validation_data_to_be_shared.csv`):** 41,792 records used for generating final default probability predictions.

### 🔑 Key Features
The dataset contains a rich set of independent variables capturing customer behavior:
- `onus_attributes`: On-us attributes like credit limit and internal bank metrics.
- `transaction_attribute`: Transaction-level attributes (number of transactions / rupee value across various merchant types).
- `bureau`: Bureau tradeline-level attributes (e.g., product holdings, historical delinquencies).
- `bureau_enquiry`: Bureau enquiry-level attributes (e.g., personal loan inquiries in the last 3 months).

## 🚀 Methodology & Machine Learning Pipeline

The machine learning solution is implemented via a rigorous end-to-end Python pipeline (`credit_card_det.py`) relying on `pandas`, `NumPy`, and `scikit-learn`.

### 1. Data Preprocessing & Cleaning
- **Missing Value Treatment:** Thoroughly checked missing data percentages. Features `bureau_436` and `bureau_447` were found to be completely empty and strategically dropped. Other numeric features underwent mean imputation using `SimpleImputer`.
- **Scaling:** Features were standardized to have a mean of 0 and a standard deviation of 1 via `StandardScaler` to stabilize models.
- **Outlier Management:** Outliers were analyzed using the Interquartile Range (IQR) method and documented. They were retained in training to capture rare default patterns but mitigated via robust scaling and tree-based modeling.

### 2. Model Development
- Explored both **Logistic Regression** (for high interpretability of feature coefficients) and **Random Forest Classifier** (for non-linear boundaries and robust handling of complex attributes).
- The primary probability generation uses a `RandomForestClassifier` yielding highly accurate probabilities of default while providing feature importance scoring.

### 3. Sanity Checks & Quality Assurance
- **Probability Normalization:** Predictions are strictly bound between `0` and `1` using `np.clip`.
- **Distribution Audits:** Built-in programmatic tests confirm the absence of NaNs/infinites and analyze prediction spreads (mean, median, std dev) before finalizing output.

## 📊 Key Insights & Observations

Based on our exploratory data analysis and model outputs:
1. **Data Anomalies:** `bureau_436` and `bureau_447` lacked any usable observations.
2. **Feature Influence:** Using Random Forest feature importance, we identified the top 15 most influential predictors driving credit default risk, documented in `feature_importance.csv`.
3. **Probability Distribution:** Analysis of the validation output revealed that **over 30,000 customers exhibit a near 0.0 probability** of defaulting. As the predicted probability of default increases, the count of customers significantly drops, though a minor spike was observed in the 0.6–0.7 probability range, allowing Bank A to narrowly target this high-risk cluster.

## 📦 Repository Structure

- **`credit_card_det.py`** — The main machine learning pipeline script.
- **`credit_card_predictions.csv`** — The final predicted probabilities against `account_number` for the validation set.
- **`feature_importance.csv`** — Ranking of the most impactful features.
- **`missing_values_analysis.csv`** & **`outlier_analysis.csv`** — Detailed data quality audit reports.
- **`Documentation of Credit Card Detection.docx`** — Deep-dive analysis and methodology write-up.

## 🛠️ Execution

To run the pipeline and generate predictions from scratch, install dependencies and execute the script:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python credit_card_det.py
```
This will automatically process the CSVs, train the ensemble model, plot data insights, and export the validation predictions.
