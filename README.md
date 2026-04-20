# March Madness Winner Prediction Pipeline

This project develops a machine learning pipeline to predict NCAA Tournament game outcomes. By leveraging historical data from 2016–2025—including advanced metrics like **KenPom**, **Barttorvik**, and **EvanMiya**—the pipeline identifies betting-edge upsets and builds robust brackets.

## 📌 Project Overview
Predicting March Madness is notoriously difficult due to the single-elimination format. This pipeline addresses the "unpredictability" by focusing on **Relative Strength Features** and **Symmetry Augmentation** using Logistic Regression and SGD Classifiers.



---

## 📊 Dataset
Data is sourced from the `nishaanamin/march-madness-data` Kaggle dataset, featuring:
* **Advanced Metrics:** KenPom, Barttorvik, TeamRankings, EvanMiya, RPPF, and Heat Check.
* **Tournament Context:** Historical matchups, seeds, and results.
* **Travel & Fatigue:** Travel distances and timezone crosses for each venue.
* **Momentum:** Pre-season vs. end-of-season rating deltas.

---

## ⚙️ The Pipeline

### 1. Data Preprocessing
* **Temporal Filtering:** Data limited to 2016+ to ensure feature consistency and minimize missing values.
* **Difference Engineering:** Rather than raw stats, the model learns from the **delta** between Team A and Team B (e.g., `ADJ_OE_DIFF`).
* **Symmetry Augmentation:** Every game is recorded twice (A vs B and B vs A) with negated features. This doubled our training data and was the **single most effective** improvement for model stability.

### 2. Model Architecture
We utilized a multi-stage approach to select and optimize models:
* **Feature Selection:** `SelectFromModel` via XGBoost to identify the top 25 most predictive metrics.
* **Hyperparameter Tuning:** Bayesian optimization via **Optuna** (50 trials per model).
* **Cross-Validation:** `TimeSeriesSplit` to prevent temporal leakage.

### 3. Models Evaluated
| Model | Test Accuracy | Note |
| :--- | :--- | :--- |
| **SGDClassifier** | **82.5%** | **Best Performer**; exceptional at identifying upsets. |
| **Logistic Regression** | 80.2% | More stable probabilities, but more conservative. |
| **Seed Baseline** | 74.6% | The "Naive" approach of picking the higher seed. |
| **Soft Ensemble** | 80.9% | Weighted average of SGD and LR. |

---

## 📈 Key Findings
* **Upset Specialists:** In the "upset-heavy" 2024 tournament, the SGD model provided a **15.9% lift** over the seed baseline.
* **The "Chalk" Limitation:** In years where favorites dominate (like 2025), the model performs on par with basic seeding.
* **Simplicity Wins:** Expanding the hyperparameter search space beyond 3-4 parameters led to overfitting due to the relatively small tournament sample size.

---

## 🚀 Usage

### Installation
```bash
pip install pandas numpy scikit-learn optuna xgboost lightgbm matplotlib seaborn kagglehub joblib
```

### Predict a Single Matchup
```python
winner, p0, p1 = predict_matchup("Duke", "Siena", year=2026, round_num=64)
print(f"Predicted winner: {winner} ({max(p0, p1):.1%} confidence)")
```

### Run a Full Bracket
```python
bracket = {
    "East": [("Duke", "Siena"), ("Ohio St.", "TCU")],
    # Add other regions...
}
champion = predict_bracket(bracket, year=2026)
```

---

## ⚠️ Caveats
* **Sample Size:** The test set (126 games) carries a confidence interval of $\pm 7.4\%$.
* **Intangibles:** The model cannot account for late-breaking injuries, locker room chemistry, or coaching changes.
* **Stochastic Nature:** Due to the `SGDClassifier`, results may vary slightly (1–2%) across different random seeds.
