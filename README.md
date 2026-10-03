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
- [Conclusion](#conclusion)
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

Results from one default run (`python -m src.train`: 5-fold CV, one seed, no tuning) on a 4-core machine, which took about 6 minutes. All scores are **out-of-fold on the 63,999 training rows**, not Kaggle leaderboard scores. Results on another machine or with other library versions may differ slightly.

| Model | ROC AUC | F1 @ 0.5 | Best threshold | F1 @ best threshold |
|---|---|---|---|---|
| **Ensemble (mean)** | **0.6751** | 0.6199 | 0.355 | 0.6759 |
| HistGradientBoosting | 0.6735 | 0.6184 | 0.350 | 0.6757 |
| XGBoost | 0.6735 | 0.6183 | 0.340 | 0.6762 |
| CatBoost | 0.6730 | 0.6182 | 0.345 | 0.6753 |
| LightGBM | 0.6724 | 0.6169 | 0.345 | 0.6754 |
| Logistic regression | 0.6687 | 0.6195 | 0.355 | 0.6741 |
| Random forest | 0.6679 | 0.6189 | 0.310 | 0.6726 |
| *Baseline: everyone responds* | – | 0.6665 | – | – |

What this shows:
- **Threshold tuning is the biggest F1 gain.** Every model goes from about 0.62 to about 0.675 F1 just by lowering the threshold to about 0.35.
- **The models only beat the "everyone responds" baseline by about 0.01 F1.** The data has a weak signal (AUC ≈ 0.67), so the F1-optimal strategy is to contact most customers (about 79% at the chosen threshold).
- **Boosting models and the ensemble rank customers best.** The ensemble has the highest AUC, but the differences between the boosting libraries are small.
- In LightGBM, the most-used features (by split count) were `Balance`, the K-Means distance features, `BalancePerProduct`, `Transaction4`, and the transaction-to-balance ratio.

These are local CV scores. The real test is submitting `outputs/submission_ensemble.csv` (or another file) to Kaggle, if late submissions are still accepted.

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

## Conclusion

### All results in one table

The table compares every model from the original notebook (v1) and the improved pipeline (v2). The **F1** column is the F1 score of the "responds" class (`Target = 1`), the class the competition scores.

| Rank | Version | Model | Tuning / threshold | Kaggle public F1 | Local F1 | Local ROC AUC |
|---|---|---|---|---|---|---|
| 1 | v2 | **XGBoost** | Defaults, threshold 0.340 | not submitted | **0.6762** | 0.6735 |
| 2 | v2 | **Ensemble (mean of 6 models)** | Threshold 0.355 | not submitted | 0.6759 | **0.6751** |
| 3 | v2 | HistGradientBoosting | Defaults, threshold 0.350 | not submitted | 0.6757 | 0.6735 |
| 4 | v2 | LightGBM | Defaults, threshold 0.345 | not submitted | 0.6754 | 0.6724 |
| 5 | v2 | CatBoost | Defaults, threshold 0.345 | not submitted | 0.6753 | 0.6730 |
| 6 | v2 | Logistic regression | Threshold 0.355 | not submitted | 0.6741 | 0.6687 |
| 7 | v2 | Random forest | Threshold 0.310 | not submitted | 0.6726 | 0.6679 |
| – | – | *Baseline: everyone responds* | – | – | *0.6665* | *0.5000* |
| 8 | v1 | GradientBoostingClassifier | Grid search, threshold 0.5 | **0.63851** | 0.6572 | – |
| 9 | v1 | CatBoostClassifier | Grid search, threshold 0.5 | 0.62837 | 0.6561 | – |
| 10 | v1 | CatBoostClassifier | Randomized search, threshold 0.5 | 0.62500 | 0.6556 | – |
| 11 | v1 | RandomForestClassifier | Grid search, threshold 0.5 | 0.61824 | 0.6367 | – |
| 12 | v1 | GradientBoostingClassifier | Randomized search, threshold 0.5 | 0.61655 | 0.6572 | – |
| 13 | v1 | LogisticRegression | Grid search, threshold 0.5 | 0.60135 | 0.6462 | – |
| 14 | v1 | RandomForestClassifier | Randomized search, threshold 0.5 | 0.59628 | 0.6369 | – |
| 15 | v1 | LogisticRegression | Randomized search, threshold 0.5 | 0.59459 | n/a ¹ | – |
| 16 | v1 | DecisionTreeClassifier | Grid / randomized search, threshold 0.5 | 0.52702 | 0.6450 / 0.6407 | – |
| – | v1 | ANN (Keras + PCA) | Manual, threshold 0.5 | not submitted | 0.6360 ² | – |

Ranking: v2 models by local F1, then v1 models by Kaggle public F1 (their only score on the real test set).

How to read the scores:
- **Kaggle public F1:** the score Kaggle gave each v1 submission, on about 40% of `test.csv`. This is the only score measured on the real test data. The v2 models have not been submitted yet.
- **Local F1, v1:** from the classification reports saved in the notebook (25% hold-out, 13,579 rows, threshold 0.5). Those outputs come from the notebook's final run, while the Kaggle scores came from earlier runs, so the two columns don't always line up.
- **Local F1 and AUC, v2:** 5-fold out-of-fold scores on all 63,999 training rows (one seed, no Optuna tuning).
- v1 and v2 local scores use different data splits, so they are **close but not exactly comparable**. Small differences (around 0.001) between v2 models are within noise.

¹ The notebook evaluated this model using the GradientBoosting predictions by mistake, so its saved F1 is not its own.
² Keras' batch-averaged F1 metric, so only approximate.

### Which model is best?

- **Best submitted model (v1):** **GradientBoostingClassifier with grid search**, with a Kaggle public F1 of **0.63851**.
- **Best model overall (v2):** the **ensemble** is the recommended choice. Its F1 (0.6759) is tied with the best single model, XGBoost (0.6762), but it has the highest AUC (0.6751), meaning it ranks customers best, and averaging several models usually makes it more stable on unseen data. XGBoost is the best single model if you want something simpler.
- **Why v2 scores higher:** mostly the **F1-optimal threshold** (about 0.35 instead of 0.5), which raised every model's F1 from about 0.62 to about 0.675. The leak-free cluster features and the new rate and transaction features add a smaller gain in AUC.
- **Honest limit:** all models beat the "everyone responds" baseline (F1 0.6665) by only about 0.01. The features carry a weak signal (AUC ≈ 0.67), so no model separates responders from non-responders very well.

**Next step:** submit `outputs/submission_ensemble.csv` and `outputs/submission_xgb.csv` to Kaggle, if late submissions are allowed, to confirm the v2 gain on the real test set. Then run `python -m src.train --tune --n-trials 50 --seeds 3` to see whether tuning adds anything.

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
