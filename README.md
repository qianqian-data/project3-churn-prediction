# Churn Prediction Model — Subscription Business

## Project Overview
Machine learning models to predict customer churn 30 days in advance,
enabling proactive retention interventions.

**Tools:** Python, pandas, scikit-learn, matplotlib  
**Dataset:** IBM Telco Customer Churn (Kaggle)  
**Best Model:** Logistic Regression (AUC = 0.832)

## Model Performance
| Model | AUC Score |
|-------|-----------|
| Logistic Regression | 0.832 |
| Random Forest | 0.819 |
| Random baseline | 0.500 |

## Top Churn Predictors
1. TotalCharges — cumulative spend signals dissatisfaction
2. MonthlyCharges — high price = high churn risk
3. Tenure — new users are most vulnerable
4. Contract type — month-to-month users are 15× more likely to churn

## Business Recommendations
1. **Flag high-risk users early** — users with high monthly charges 
   and tenure under 6 months are the highest priority
2. **Target contract upgrades** — use model scores to prioritize 
   month-to-month users for upgrade campaigns
3. **Proactive support outreach** — users without TechSupport or 
   OnlineSecurity are more likely to churn

## Business Impact
Deploying this model to flag the top 20% highest-risk users could 
identify ~374 at-risk customers per month before they leave.

## Files
- `analysis.ipynb` — full modeling notebook
- `data/telco-churn.csv` — dataset
- `charts/` — ROC curve and feature importance plots

## How to Run
```bash
pip install pandas matplotlib scikit-learn
jupyter notebook analysis.ipynb
```