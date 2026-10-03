# Target Marketing for a Canadian Bank

Binary classification project that predicts which existing bank customers are most likely to respond to a deposit marketing campaign. It was built for the Kaggle in-class competition [Target Marketing for Canadian Bank (AML-1413, Winter 2023)](https://www.kaggle.com/competitions/target-marketing-for-canadian-bank-2023w-aml1413).

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-yellow)
![LightGBM](https://img.shields.io/badge/LightGBM%20%7C%20XGBoost-Boosting-green)
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
- [Improved pipeline (v2)](#improved-pipeline-v2)
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

This section describes the original competition submission in [`target_marketing_canadian_bank.ipynb`](target_marketing_canadian_bank.ipynb). A cleaner, improved version is described in [Improved pipeline (v2)](#improved-pipeline-v2).

### 1. Exploratory data analysis
- Distributions, box plots, and correlation heatmaps for every numeric column.
- Most continuous features (for example `Balance` and the transactions) are strongly right-skewed.
- Products, transactions, activity indicators, and previous campaign results are correlated with each other, and the competitive rates are correlated with `RateBefore` and `ReferenceRate`. These groups shaped the feature engineering.

### 2. Feature engineering
- **Aggregate features:** total products owned, sum of transactions, number of external accounts, and average competitive rate.
- **K-Means cluster features:** the number of clusters for each feature combination was chosen with the elbow method (Yellowbrick `KElbowVisualizer`), for example:
  - `Balance` + sum of transactions
  - `Balance` + `Product1`
  - `Transaction4` + `Product2` + `Transaction3`
  - Pairs of transactions, such as `Transaction4` + `Transaction5`
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
- **The models are only moderately good at finding responders.** False negatives stayed high, which directly lowers F1. More feature work or threshold tuning would be needed before using this in a real campaign.

## Improved pipeline (v2)

After reviewing the original notebook, I rebuilt the modelling as a reproducible script in [`src/`](src/). The main changes fix problems in the original notebook.

### Issues found in the original notebook

| Issue | Effect | Fix in v2 |
|---|---|---|
| K-Means and `StandardScaler` were fitted **separately** on train and test | Cluster *k* in the test set was not the same group as cluster *k* in the training set, so the cluster features were inconsistent at prediction time | Fitted on the training folds only, then applied to validation and test data (`ClusterFeatures` transformer) |
| `ActivityIndicator` (a count with 611 distinct values) was one-hot encoded | About 600 sparse columns, and test values unseen in training were silently set to 0 | Kept as a number, with a log transform and ratio features |
| Thresholds were left at 0.5 and models were tuned for ROC AUC or accuracy | The competition metric is F1, which is very sensitive to the threshold | The threshold is chosen to maximise F1 on out-of-fold predictions |
| Models were scored on one 75/25 split and the final model saw only 75% of the data | Noisy scores, and less data for the submission | Stratified 5-fold CV, with test predictions averaged across folds |
| `feature_importance()` plotted `argsort()` ranks instead of importances | The feature-importance charts and their conclusions are not reliable | Real importances are saved to `outputs/feature_importance_<model>.csv` |
| The logistic regression randomized-search metrics used the GradientBoosting predictions (`gbr_y_pred`) | Those reported metrics belong to another model | Every model is evaluated on its own predictions |
| `PreviousCampaignResult` is always 0 in `test.csv` | The feature cannot help the test predictions | Dropped |

The original notebook is kept unchanged, because its saved outputs and Kaggle scores came from that exact code.

### What v2 adds

- **More features:** transaction statistics (positive/negative sums, counts, max, standard deviation), log transforms, balance ratios, and rate features that describe how good the offer is compared with what the customer has, such as `MaxCompetitiveRate - RateBefore` and the number of offers above `RateBefore`.
- **More models:** logistic regression, random forest, extra trees, scikit-learn HistGradientBoosting, LightGBM, XGBoost, and CatBoost.
- **Optional Optuna tuning** for LightGBM, XGBoost, and CatBoost (`--tune`).
- **Ensemble:** the average of all models' probabilities, with its own F1-optimal threshold.
- **Seed averaging** (`--seeds N`) for more stable scores.

### A note about F1 on this dataset

The training set is balanced: 50.0% of customers responded. With F1, predicting that **every** customer responds already gives precision ≈ 0.50 and recall = 1.0, so F1 ≈ **0.667**. That is higher than the original best public score (0.63851), because the original models used a 0.5 threshold, which trades recall for precision.

So v2 reports **ROC AUC** to show how well each model ranks customers, and **F1 at the tuned threshold** for the competition score. The script prints the "everyone responds" baseline so you can see how much the model actually adds. In a real campaign, the useful output is the ranking of customers by probability, not the F1 number.

### v2 results

Run `python -m src.train` to reproduce. The out-of-fold results table is printed at the end and saved to `outputs/results.csv`.

### Running v2

```bash
pip install pandas scikit-learn lightgbm xgboost catboost optuna

python -m src.train                                   # 6 models, 5-fold CV
python -m src.train --tune --n-trials 50 --seeds 3    # tuned and seed-averaged (slower)
python -m src.train --models lgbm catboost            # only some models
python -m src.train --help                            # all options
```

Results go to `outputs/`:

| File | Content |
|---|---|
| `results.csv` | Out-of-fold AUC, log loss, accuracy, F1 at 0.5, best threshold, and F1 at that threshold for each model |
| `submission_<model>.csv`, `submission_ensemble.csv` | Kaggle submission files (thresholded at the F1-optimal value) |
| `oof_predictions.csv`, `test_probabilities.csv` | Probabilities, for stacking or further analysis |
| `feature_importance_<model>.csv` | Feature importances of the tree models |
| `best_params.json` | Optuna's best parameters (with `--tune`) |

The original submissions used `Customer_id` as the ID column header, while the competition page shows `CustomerID`. If Kaggle rejects the file, run with `--id-header CustomerID`.

## Repository structure

```
.
├── data/
│   ├── train.csv                          # Labelled training data
│   └── test.csv                           # Unlabelled data to score
├── src/
│   ├── features.py                        # v2 feature engineering (leak-free K-Means clusters)
│   └── train.py                           # v2 CV training, threshold tuning, ensemble, submissions
├── target_marketing_canadian_bank.ipynb   # Original competition notebook: EDA, features, models, results
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

Ideas not yet in v2, roughly in order of expected value:

- **Stacking:** train a logistic regression on `outputs/oof_predictions.csv` instead of a plain average.
- **Weighted ensemble:** search for blend weights on the out-of-fold predictions.
- **Feature selection:** drop features with near-zero importance, or use permutation importance to find features that hurt.
- **Target encoding** of `ActivityIndicator` and `RegularInteractionIndicator`, fitted inside each fold.
- **Adversarial validation:** check whether `test.csv` looks different from `train.csv` (for example, `PreviousCampaignResult` is always 0 in test), and validate on the most test-like training rows.
- **Business view:** report lift and gains charts (responders found in the top 10%, 20%, … of customers ranked by probability), which matter more for a real campaign than F1.

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
