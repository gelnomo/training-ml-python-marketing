# Target Marketing for a Canadian Bank

Binary classification project that predicts which existing bank customers are most likely to respond to a deposit marketing campaign. It was built for the Kaggle in-class competition [Target Marketing for Canadian Bank (AML-1413, Winter 2023)](https://www.kaggle.com/competitions/target-marketing-for-canadian-bank-2023w-aml1413).

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-yellow)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-red)

**Lead author:** Erwin Mongui, main contributor behind this work ([see Team](#team))

| Competition | Details |
|---|---|
| Host | Kaggle in-class competition, AML-1413 Winter 2023 |
| Dates | Mar 13, 2023 – Apr 17, 2023 |
| Task | Binary classification (`Target`) |
| Evaluation metric | **F1 score** |
| Best public score | **0.63851** (GradientBoostingClassifier) |

---

## Table of contents

- [Competition](#competition)
- [Business problem](#business-problem)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Key takeaways](#key-takeaways)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Future work](#future-work)
- [Team](#team)
- [Citation](#citation)

---

## Competition

- **Competition page:** https://www.kaggle.com/competitions/target-marketing-for-canadian-bank-2023w-aml1413
- **Metric:** F1 score on the `Target` column.
- **Leaderboard:** the public leaderboard used about 40% of the test data, and the final private standings used the other 60%.
- **Submission format:** a CSV file with a header and one row per test customer:

  ```
  CustomerID,Target
  CTSP######,1
  ```

## Business problem

Bank "A" wants to run a marketing campaign that offers existing clients promotional interest rates to attract deposit balances. Contacting every customer is expensive, so the task is to **find the right set of customers who are most likely to respond** to the campaign.

This is a supervised binary classification problem: given a customer's profile, predict `Target` (1 = responds to the campaign, 0 = does not). Because the metric is F1, the model has to find responders (recall) without flagging too many non-responders (precision).

## Dataset

The data comes from a real international bank and has been de-identified to protect customer and bank privacy.

| File | Rows | Description |
|---|---|---|
| `data/train.csv` | 63,999 | Customers from previous campaigns, with the `Target` label |
| `data/test.csv` | 1,480 | Customers to score, without labels |

| Column(s) | Description |
|---|---|
| `Customer_id` | Anonymous ID unique to each customer |
| `Target` | 1 if the customer responded to the campaign (train only) |
| `Balance` | Credit card balance |
| `PreviousCampaignResult` | Result of the last campaign |
| `Product1` … `Product6` | Different products owned by the customer |
| `Transaction1` … `Transaction9` | Last 9 transactions made by the customer |
| `ExternalAccount1` … `ExternalAccount7` | External accounts held by the customer, including bills |
| `ActivityIndicator` | Number of activities done by the customer, including telebanking and going to the bank |
| `RegularInteractionIndicator` | How often the customer interacts with the bank (rating scale) |
| `CompetitiveRate1` … `CompetitiveRate7` | Interest rates offered to the customer |
| `RateBefore` | Interest rates the customer had on the products before the competitive rates were offered |
| `ReferenceRate` | Interest rates the customer had on the products after negotiating the competitive rates |

Field descriptions come from the competition's data page. Neither file has missing or duplicated values.

> **Data license:** the data is "Subject to Competition Rules" on Kaggle. It is included here for reproducibility. See the competition's [Rules](https://www.kaggle.com/competitions/target-marketing-for-canadian-bank-2023w-aml1413/rules) page before reusing it.

## Approach

The full pipeline is in [`target_marketing_canadian_bank.ipynb`](target_marketing_canadian_bank.ipynb).

### 1. Exploratory data analysis
- Distributions, box plots, and correlation heatmaps for every numeric column.
- Most continuous features (for example `Balance` and the transactions) are strongly right-skewed.
- Products, transactions, activity indicators, and previous campaign results are correlated with each other, and the competitive rates are correlated with `RateBefore` and `ReferenceRate`. These groups shaped the feature engineering.

### 2. Feature engineering
- **Aggregate features:** total products owned, sum of transactions, number of external accounts, and average competitive rate.
- **K-Means cluster features:** the number of clusters for each feature combination was chosen with the elbow method (Yellowbrick `KElbowVisualizer`), for example:
  - `Balance` + sum of transactions
  - `Balance` + `PreviousCampaignResult`
  - `RateBefore` + average interest rate
  - `Balance` + products
  - Total products + sum of transactions + `ActivityIndicator`
- **One-hot encoding** of discrete columns such as `ActivityIndicator`, `RegularInteractionIndicator`, and `PreviousCampaignResult`.
- **Scaling** of large-valued columns, and dropping columns that the new aggregates replace.

### 3. Outlier handling
Several outlier-removal methods were tested (Z-score, robust Z-score, IQR, winsorization, DBSCAN, and Isolation Forest). Every removal lowered the score, so the final model keeps all rows.

### 4. Modelling
The data was split 75/25 (stratified), and each model was tuned with **GridSearchCV** and **RandomizedSearchCV**:

- CatBoostClassifier
- GradientBoostingClassifier
- LogisticRegression
- DecisionTreeClassifier
- RandomForestClassifier
- Artificial neural network (Keras), trained on PCA-reduced features

Each model was evaluated with a confusion matrix, accuracy, precision, recall, F1-score, and ROC AUC. Feature importance was inspected for the tree-based models.

## Results

Scores as recorded in the notebook. "Kaggle F1" is the public leaderboard F1 score, computed on about 40% of the test data. "Local score" is the score on the 25% hold-out split.

| Model | Tuning | Kaggle F1 (public) | Local score |
|---|---|---|---|
| **GradientBoostingClassifier** | Grid search | **0.63851** | 0.62824 |
| CatBoostClassifier | Grid search | 0.62837 | 0.62910 |
| CatBoostClassifier | Randomized search | 0.62500 | 0.62698 |
| RandomForestClassifier | Grid search | 0.61824 | 0.62206 |
| GradientBoostingClassifier | Randomized search | 0.61655 | 0.62349 |
| LogisticRegression | Grid search | 0.60135 | 0.60536 |
| RandomForestClassifier | Randomized search | 0.59628 | 0.61868 |
| LogisticRegression | Randomized search | 0.59459 | 0.62199 |
| DecisionTreeClassifier | Grid / randomized search | 0.52702 | 0.57762 |
| ANN (Keras + PCA) | Manual | not submitted | 0.6248 accuracy |

**Best submission:** GradientBoostingClassifier with grid search, with an F1 score of **0.63851** on the public leaderboard.

## Key takeaways

- **Data understanding mattered more than model choice.** Gradient-boosted trees, random forests, logistic regression, and the neural network all landed between about 0.60 and 0.64. The engineered features moved the score more than switching algorithms did.
- **Boosting models performed best.** GradientBoosting and CatBoost were consistently at the top.
- **Removing outliers hurt the score.** The "outliers" carried real signal about customer behaviour.
- **Activity-indicator features were the most important** in the tree-based models.
- **The models are only moderately good at finding responders.** False negatives stayed high, which directly lowers F1. More feature work or threshold tuning would be needed before using this in a real campaign.

## Repository structure

```
.
├── data/
│   ├── train.csv                          # Labelled training data
│   └── test.csv                           # Unlabelled data to score
├── target_marketing_canadian_bank.ipynb   # Main notebook: EDA, feature engineering, models, results
├── exploration_early_experiments.ipynb    # Early EDA and first model experiments
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/gelnomo/training-ml-python-marketing.git
cd training-ml-python-marketing

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook target_marketing_canadian_bank.ipynb
```

The notebook reads the data from the Kaggle input folder when it runs on Kaggle, and from the local `data/` folder otherwise.

> The original notebook ran on Kaggle with Python 3.7. The full run, including the hyperparameter searches, took over two hours there, so expect a long first run. `requirements.txt` is unpinned, so very recent library versions may need small code changes.

## Future work

- Tune the decision threshold to maximise F1 directly (the models currently use the default 0.5), or use class weights to reduce false negatives.
- Try LightGBM or XGBoost, and stack the best boosting models.
- Use cross-validated target encoding instead of one-hot encoding for high-cardinality indicators.
- Move the notebook logic into reusable Python modules with a reproducible training script.

## Team

This project was submitted by **Group Theta** for the AML-1413 (Winter 2023) in-class competition.

| Member | Role |
|---|---|
| **Erwin Mongui (883945)** | **Lead author, main contributor behind this work** |
| Dev Makwana (885064) | Team member |
| Aparna Manjalavil (887001) | Team member |
| Yasar Mansuri (884738) | Team member |
| Amarnadh Movva (884107) | Team member |

## Citation

Competition:

> Ashish Gupta. *Target Marketing for Canadian Bank*. https://www.kaggle.com/competitions/target-marketing-for-canadian-bank-2023w-aml1413, 2023. Kaggle.
