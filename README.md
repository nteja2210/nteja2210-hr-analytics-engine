<<<<<<< HEAD
# Intelligent HR Analytics Engine

Special Project (3rd year) - Application area: **Workforce Analytics**

An end-to-end machine learning system that helps HR teams understand and reduce employee attrition.

| Capability | Technique | File |
|---|---|---|
| Attrition prediction | Logistic Regression, Random Forest, Gradient Boosting (compared) | `src/train.py` |
| Explainable AI | SHAP (global + per-employee) | `src/explain.py` |
| Employee segmentation | KMeans + elbow/silhouette | `src/segment.py` |
| Dashboard | Streamlit + Plotly | `app.py` |

## Project structure
```
hr-analytics-engine/
├── data/hr_data.csv        # dataset (synthetic IBM-style, or the real Kaggle file)
├── models/                 # saved model + metrics.json
├── reports/figures/        # ROC curve, confusion matrix (for your report)
├── src/
│   ├── generate_data.py    # creates synthetic dataset
│   ├── preprocess.py       # cleaning, encoding, scaling, train/test split
│   ├── train.py            # model training and comparison
│   ├── explain.py          # SHAP explanations
│   └── segment.py          # KMeans clustering
├── app.py                  # Streamlit dashboard
└── requirements.txt
```

## How to run (VS Code terminal)
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

python -m src.generate_data    # skip if you use the real Kaggle CSV
python -m src.train            # trains models, saves best one
streamlit run app.py           # opens the dashboard in your browser
```

## Using the real dataset
Download **IBM HR Analytics Employee Attrition & Performance** from Kaggle, save it as
`data/hr_data.csv` (replace the synthetic file), then run `python -m src.train` again.
Column names are identical, so no code changes are needed.

## Dashboard pages
1. **Overview** - headcount, attrition rate, average income, top 10 at-risk current employees
2. **Attrition Analysis** - attrition by overtime, department, job role, age group, income; SHAP driver chart
3. **Predict & Explain** - enter employee details, get risk %, SHAP explanation and HR action suggestions
4. **Employee Segments** - KMeans segments with business names and attrition by segment
5. **Model Performance** - model comparison table, ROC curves, confusion matrix

## Method summary
- **Preprocessing:** drop constant/ID columns, StandardScaler for numeric, one-hot for categorical, stratified 80/20 split.
- **Imbalance:** only ~16-18% of employees leave, so `class_weight="balanced"` is used and models are judged on recall, F1 and ROC-AUC, not just accuracy. The decision threshold is tuned for F1.
- **Explainability:** SHAP shows which factors raise or lower each employee's risk.
- **Segmentation:** KMeans on age, income, tenure, satisfaction, work-life balance and promotion gap; clusters are named from their centroid profile.
=======
# hr-analytics-engine
>>>>>>> 57f07e312b7186309b103b72c3b267bdd74c6d8e
# nteja2210-hr-analytics-engine
