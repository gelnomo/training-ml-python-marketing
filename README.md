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
| Best public score (original submission) | **0.63851** (GradientBoostingClassifier) |
| Best local score (improved pipeline) | **0.6770** F1, 5-fold CV (stacking ensemble) |

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
- **8 models:** logistic regression, a neural network (scikit-learn MLP), random forest, extra trees, scikit-learn HistGradientBoosting, LightGBM, XGBoost, and CatBoost.
- **Optuna tuning** (`--tune`) for LightGBM, XGBoost, CatBoost, HistGradientBoosting, and logistic regression. The search ranges include small, strongly regularised trees, which work best on this data. The tuned parameters are saved in [`configs/best_params.json`](configs/best_params.json), so you can reuse them without tuning again (`--params-file`).
- **4 ensembles:** plain average, rank average, a weighted blend (weights chosen to minimise log loss), and stacking (a logistic regression trained on the models' out-of-fold predictions and scored with nested cross-validation).
- **Reports:** the AUC standard deviation across folds (to show what counts as noise) and a lift table.
- **Options:** seed averaging (`--seeds N`), and `--no-clusters` to skip the K-Means features.

### A note about F1 on this dataset

The training set is balanced: 50.0% of customers responded. With F1, predicting that **every** customer responds already gives precision ≈ 0.50 and recall = 1.0, so F1 ≈ **0.667**. That is higher than the original best public score (0.63851), because the original models used a 0.5 threshold, which trades recall for precision.

So v2 reports **ROC AUC** to show how well each model ranks customers, and **F1 at the tuned threshold** for the competition score. The script prints the "everyone responds" baseline so you can see how much the model actually adds. In a real campaign, the useful output is the ranking of customers by probability (see the [lift table](#lift-who-to-contact-first)), not the F1 number.

### Experiments: what helped and what didn't

All scores are 5-fold out-of-fold ROC AUC on the 63,999 training rows, using LightGBM unless noted. The AUC varies by about **±0.003 between folds**, so differences below about 0.002 are noise.

| Experiment | ROC AUC | Verdict |
|---|---|---|
| Raw columns only (no feature engineering) | 0.6725 | Baseline |
| + engineered features (ratios, rates, transaction stats) | 0.6728 | No real gain for boosting models |
| + K-Means cluster features | 0.6722 | No gain for boosting models; slower |
| Slower learning (lr 0.01, 2,000 trees) | 0.6726 | No gain |
| **Small trees + strong regularisation** | **0.6747** | **Helps**, so tuning searched in this direction |
| **Optuna-tuned LightGBM (40 trials)** | **0.6752** | **Helps** (+0.003 over defaults) |
| Logistic regression **with** clusters vs **without** | 0.6687 vs 0.6637 | Clusters **help linear models** (+0.005) |
| Adversarial validation (can a model tell train from test?) | 0.498 | No: train and test look alike, so CV scores should carry over to Kaggle |

Main lessons: the signal in this data is weak and mostly already in the raw columns. Tuning toward simpler, regularised trees and combining models gives small but consistent gains. More feature engineering on its own did not.

### v2 results (tuned)

Command: `python -m src.train --tune --n-trials 40 --no-clusters` (5-fold CV, one seed, on a 4-core machine; about 40 minutes in total). All scores are **out-of-fold on the 63,999 training rows**, not Kaggle leaderboard scores. Results on another machine or with other library versions may differ slightly.

| Model | ROC AUC (± fold std) | F1 @ 0.5 | Best threshold | F1 @ best threshold |
|---|---|---|---|---|
| **Ensemble: stacking** | 0.6757 | 0.6187 | 0.350 | **0.6770** |
| **Ensemble: weighted blend** | **0.6758** | 0.6188 | 0.350 | 0.6768 |
| CatBoost (tuned) | 0.6756 ± 0.0030 | 0.6185 | 0.350 | 0.6765 |
| LightGBM (tuned) | 0.6752 ± 0.0031 | 0.6195 | 0.355 | 0.6768 |
| XGBoost (tuned) | 0.6752 ± 0.0027 | 0.6187 | 0.345 | 0.6763 |
| Ensemble: mean | 0.6749 | 0.6203 | 0.350 | 0.6762 |
| Ensemble: rank mean | 0.6748 | 0.6291 | 0.200 ¹ | 0.6765 |
| HistGradientBoosting (tuned) | 0.6740 ± 0.0027 | 0.6184 | 0.345 | 0.6757 |
| Random forest | 0.6679 ± 0.0033 | 0.6189 | 0.340 | 0.6725 |
| Logistic regression (tuned) | 0.6637 ± 0.0030 | 0.6143 | 0.360 | 0.6712 |
| MLP neural network | 0.6563 ± 0.0030 | 0.6142 | 0.320 | 0.6699 |
| *Baseline: everyone responds* | 0.5000 | 0.6665 | – | – |

¹ The rank mean outputs ranks between 0 and 1, not probabilities, so its threshold isn't comparable to the others.

The weighted blend gave CatBoost 45% of the weight, LightGBM 19%, HistGradientBoosting 17%, XGBoost 13%, and the rest about 7% in total.

What this shows:
- **Threshold tuning is the biggest F1 gain.** Every model goes from about 0.62 to about 0.676 F1 just by lowering the threshold to about 0.35.
- **Tuning added about +0.002 to +0.003 AUC** to the boosting models (for example, CatBoost went from 0.6730 to 0.6756). This is small but consistent.
- **The neural network and logistic regression are clearly weaker** on this tabular data.
- **The models only beat the "everyone responds" baseline by about 0.01 F1.** The data has a weak signal (AUC ≈ 0.675), so the F1-optimal strategy is to contact most customers (about 80% at the chosen threshold).

#### Lift: who to contact first

The ranking is more useful for the business than F1. For the weighted blend, ranking customers by predicted probability gives:

| Customers contacted (top %) | Response rate | Share of all responders found | Lift vs random |
|---|---|---|---|
| 10% | 75.1% | 15.0% | 1.50× |
| 20% | 72.4% | 29.0% | 1.45× |
| 30% | 69.3% | 41.6% | 1.39× |
| 50% | 63.1% | 63.1% | 1.26× |
| 100% | 50.0% | 100% | 1.00× |

So if the bank can only call 10% of customers, the model's top picks respond 75% of the time, compared with 50% for a random selection.

### Running v2

```bash
pip install pandas scikit-learn lightgbm xgboost catboost optuna

python -m src.train                                               # 7 models, default parameters, about 6 min
python -m src.train --params-file configs/best_params.json --no-clusters --seeds 3   # tuned parameters, 3 seeds
python -m src.train --tune --n-trials 100 --no-clusters           # tune again from scratch (slow)
python -m src.train --models lgbm catboost xgb                    # only some models
python -m src.train --help                                        # all options
```

Results go to `outputs/`:

| File | Content |
|---|---|
| `results.csv` | Out-of-fold AUC (and its fold standard deviation), log loss, accuracy, F1 at 0.5, best threshold, and F1 at that threshold for each model and ensemble |
| `submission_<model>.csv`, `submission_ensemble_<type>.csv` | Kaggle submission files (thresholded at the F1-optimal value) |
| `lift.csv` | Lift table for the best model or ensemble |
| `oof_predictions.csv`, `test_probabilities.csv` | Probabilities, for further stacking or analysis |
| `feature_importance_<model>.csv` | Feature importances of the tree models |
| `best_params.json` | Parameters used (from `--params-file` and/or `--tune`) |

The original submissions used `Customer_id` as the ID column header, while the competition page shows `CustomerID`. If Kaggle rejects the file, run with `--id-header CustomerID`.

## Conclusion

### All results in one table

The table compares every model from the original notebook (v1) and the improved pipeline (v2). The **F1** column is the F1 score of the "responds" class (`Target = 1`), the class the competition scores.

| Rank | Version | Model | Tuning / threshold | Kaggle public F1 | Local F1 | Local ROC AUC |
|---|---|---|---|---|---|---|
| 1 | v2 | **Ensemble: stacking** | Tuned models, threshold 0.350 | not submitted | **0.6770** | 0.6757 |
| 2 | v2 | **Ensemble: weighted blend** | Tuned models, threshold 0.350 | not submitted | 0.6768 | **0.6758** |
| 3 | v2 | LightGBM | Optuna, threshold 0.355 | not submitted | 0.6768 | 0.6752 |
| 4 | v2 | CatBoost | Optuna, threshold 0.350 | not submitted | 0.6765 | 0.6756 |
| 5 | v2 | Ensemble: rank mean | Tuned models, threshold 0.200 | not submitted | 0.6765 | 0.6748 |
| 6 | v2 | XGBoost | Optuna, threshold 0.345 | not submitted | 0.6763 | 0.6752 |
| 7 | v2 | Ensemble: mean | Tuned models, threshold 0.350 | not submitted | 0.6762 | 0.6749 |
| 8 | v2 | HistGradientBoosting | Optuna, threshold 0.345 | not submitted | 0.6757 | 0.6740 |
| 9 | v2 | Random forest | Defaults, threshold 0.340 | not submitted | 0.6725 | 0.6679 |
| 10 | v2 | Logistic regression | Optuna, threshold 0.360 | not submitted | 0.6712 | 0.6637 |
| 11 | v2 | MLP neural network | Defaults, threshold 0.320 | not submitted | 0.6699 | 0.6563 |
| – | – | *Baseline: everyone responds* | – | – | *0.6665* | *0.5000* |
| 12 | v1 | GradientBoostingClassifier | Grid search, threshold 0.5 | **0.63851** | 0.6572 | – |
| 13 | v1 | CatBoostClassifier | Grid search, threshold 0.5 | 0.62837 | 0.6561 | – |
| 14 | v1 | CatBoostClassifier | Randomized search, threshold 0.5 | 0.62500 | 0.6556 | – |
| 15 | v1 | RandomForestClassifier | Grid search, threshold 0.5 | 0.61824 | 0.6367 | – |
| 16 | v1 | GradientBoostingClassifier | Randomized search, threshold 0.5 | 0.61655 | 0.6572 | – |
| 17 | v1 | LogisticRegression | Grid search, threshold 0.5 | 0.60135 | 0.6462 | – |
| 18 | v1 | RandomForestClassifier | Randomized search, threshold 0.5 | 0.59628 | 0.6369 | – |
| 19 | v1 | LogisticRegression | Randomized search, threshold 0.5 | 0.59459 | n/a ¹ | – |
| 20 | v1 | DecisionTreeClassifier | Grid / randomized search, threshold 0.5 | 0.52702 | 0.6450 / 0.6407 | – |
| – | v1 | ANN (Keras + PCA) | Manual, threshold 0.5 | not submitted | 0.6360 ² | – |

Ranking: v2 models by local F1, then v1 models by Kaggle public F1 (their only score on the real test set).

How to read the scores:
- **Kaggle public F1:** the score Kaggle gave each v1 submission, on about 40% of `test.csv`. This is the only score measured on the real test data. The v2 models have not been submitted yet.
- **Local F1, v1:** from the classification reports saved in the notebook (25% hold-out, 13,579 rows, threshold 0.5). Those outputs come from the notebook's final run, while the Kaggle scores came from earlier runs, so the two columns don't always line up.
- **Local F1 and AUC, v2:** 5-fold out-of-fold scores on all 63,999 training rows, from the tuned run above.
- v1 and v2 local scores use different data splits, so they are **close but not exactly comparable**. Differences of about 0.001–0.002 between v2 models are within noise.

¹ The notebook evaluated this model using the GradientBoosting predictions by mistake, so its saved F1 is not its own.
² Keras' batch-averaged F1 metric, so only approximate.

### Which model is best?

- **Best submitted model (v1):** **GradientBoostingClassifier with grid search**, with a Kaggle public F1 of **0.63851**.
- **Best model overall (v2):** the **stacking or weighted-blend ensemble of the tuned models**. They have the best F1 (0.6770 and 0.6768) and the best AUC (0.6757 and 0.6758). The two are within noise of each other; the weighted blend is simpler to explain.
- **Best single model:** **tuned CatBoost** (AUC 0.6756) or **tuned LightGBM** (F1 0.6768). LightGBM is about 2.5 times faster.
- **Why v2 scores higher:** mostly the **F1-optimal threshold** (about 0.35 instead of 0.5), which raised every model's F1 from about 0.62 to about 0.676. Tuning toward small, regularised trees and combining models add a smaller, consistent gain in AUC.
- **Honest limit:** all models beat the "everyone responds" baseline (F1 0.6665) by only about 0.01. The features carry a weak signal (AUC ≈ 0.675), so no model separates responders from non-responders very well. The lift table shows where the model is still useful: its top 10% of customers respond 75% of the time, against 50% at random.

**Next step:** submit `outputs/submission_ensemble_weighted.csv`, `outputs/submission_ensemble_stacking.csv`, and `outputs/submission_lgbm.csv` to Kaggle, if late submissions are allowed, to confirm the v2 gain on the real test set.

## Repository structure

```
.
├── configs/
│   └── best_params.json                   # Optuna-tuned parameters (reuse with --params-file)
├── data/
│   ├── train.csv                          # Labelled training data
│   └── test.csv                           # Unlabelled data to score
├── src/
│   ├── features.py                        # v2 feature engineering (leak-free K-Means clusters)
│   └── train.py                           # v2 CV training, tuning, threshold, ensembles, submissions
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

Ideas not yet tested, roughly in order of expected value:

- **Use the clusters for the linear models only:** they added +0.005 AUC to logistic regression but nothing to the boosting models.
- **Longer tuning with seed averaging:** `--n-trials 100 --seeds 3` would give more stable parameters and scores.
- **Target encoding** of `ActivityIndicator` and `RegularInteractionIndicator`, fitted inside each fold.
- **Feature selection:** drop features with near-zero importance, or use permutation importance to find features that hurt.
- **Different metric for tuning:** tune directly for F1 at the best threshold instead of AUC (riskier, because F1 is noisier).

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
