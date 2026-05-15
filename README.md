# Heart Disease Prediction

Certification Project III — Edureka Data Science & Machine Learning Program

---

The dataset has 253,680 records from a CDC health survey, where only ~9% of respondents had heart disease. That imbalance was the main challenge — a model that just predicts "no disease" every time would hit 91% accuracy but be useless in practice.

I trained 9 model variants across 5 algorithms, tried three different ways to handle the imbalance (SMOTE, undersampling, class weights), and tuned the decision threshold using the Precision-Recall curve to prioritise recall over raw accuracy.

---

## What I built

- EDA across all 15 features — distributions, correlations, disease rates by group
- 4 engineered features: `RiskScore`, `BMI_Category`, `PoorHealth`, `UnhealthyLifestyle`
- Trained and compared: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN
- Sklearn/imblearn pipeline for clean end-to-end inference

---

## Results

| Model | ROC-AUC | Recall | F1 |
|---|---|---|---|
| Gradient Boosting (original) | **0.8105** | 0.033 | 0.062 |
| Random Forest (balanced) | 0.8065 | 0.768 | **0.370** |
| Logistic Regression (balanced) | 0.8060 | 0.775 | 0.369 |
| LR (Undersampling) | 0.8055 | **0.890** | 0.325 |
| LR (SMOTE) | 0.8014 | 0.880 | 0.323 |

Gradient Boosting had the best ROC-AUC but almost zero recall — it was essentially ignoring the positive class. For a health risk problem, missing actual cases is worse than a false alarm, so the final pipeline uses Random Forest + SMOTE which gives a better recall/precision balance.

**Top 5 features driving heart disease risk:**
1. Age
2. High Blood Pressure
3. BMI
4. RiskScore (engineered)
5. High Cholesterol

---

## Charts

| | |
|---|---|
| ![Target Distribution](images/target_distribution.png) | ![Feature Correlation](images/feature_correlation.png) |
| ![Model Comparison](images/model_comparison.png) | ![Feature Importance](images/feature_importance.png) |
| ![Confusion Matrix](images/confusion_matrix_best.png) | ![Precision-Recall Curve](images/precision_recall_curve.png) |
| ![Resampling Comparison](images/resampling_comparison.png) | ![Disease Rate by Feature](images/target_by_binary.png) |

---

## Run it

```bash
pip install -r requirements.txt
jupyter notebook heart_disease_prediction.ipynb
```

Place `heartdisease.csv` in a `data/` folder first — the file isn't tracked here due to size.

---

## Stack

Python · pandas · scikit-learn · imbalanced-learn · matplotlib · seaborn
