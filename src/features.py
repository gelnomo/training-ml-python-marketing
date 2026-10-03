"""
Feature engineering for the Target Marketing for Canadian Bank competition.

Everything that learns from data (scaling, K-Means) lives inside a scikit-learn
transformer, so it is fitted on the training folds only and then *applied* to the
validation fold and the test set. The original notebook fitted K-Means and the
scaler separately on train and test, so cluster ids did not mean the same thing in
both sets.
"""
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ID_COLUMN = "Customer_id"
TARGET_COLUMN = "Target"

PRODUCT_COLUMNS = [f"Product{i}" for i in range(1, 7)]
TRANSACTION_COLUMNS = [f"Transaction{i}" for i in range(1, 10)]
EXTERNAL_ACCOUNT_COLUMNS = [f"ExternalAccount{i}" for i in range(1, 8)]
COMPETITIVE_RATE_COLUMNS = [f"CompetitiveRate{i}" for i in range(1, 8)]

# PreviousCampaignResult is always 0 in test.csv (and non-zero for only 93 training
# rows), so it cannot help the test predictions.
DROP_COLUMNS = [ID_COLUMN, "PreviousCampaignResult"]


def signed_log1p(values: pd.Series) -> pd.Series:
    """
    Log transform that keeps the sign, for heavy-tailed columns with negative values
    :param values: numeric values
    :return: transformed values
    """
    return np.sign(values) * np.log1p(np.abs(values))


def add_row_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that only depend on the values of the same row
    :param data: raw dataframe
    :return: dataframe with the new features
    """
    df = data.copy()
    transactions = df[TRANSACTION_COLUMNS]
    rates = df[COMPETITIVE_RATE_COLUMNS]

    # Products and external accounts
    df["TotalProducts"] = df[PRODUCT_COLUMNS].sum(axis=1)
    df["TotalExternalAccount"] = df[EXTERNAL_ACCOUNT_COLUMNS].sum(axis=1)
    df["ProductsMinusExternal"] = df["TotalProducts"] - df["TotalExternalAccount"]

    # Transactions
    df["SumTransactions"] = transactions.sum(axis=1)
    df["SumPositiveTransactions"] = transactions.clip(lower=0).sum(axis=1)
    df["SumNegativeTransactions"] = transactions.clip(upper=0).sum(axis=1)
    df["CountPositiveTransactions"] = transactions.gt(0).sum(axis=1)
    df["CountNegativeTransactions"] = transactions.lt(0).sum(axis=1)
    df["CountTransactions"] = transactions.ne(0).sum(axis=1)
    df["MaxTransaction"] = transactions.max(axis=1)
    df["StdTransactions"] = transactions.std(axis=1)
    df["AverageTransactions"] = df["SumTransactions"] / df["CountTransactions"].replace(0, np.nan)
    df["AverageTransactions"] = df["AverageTransactions"].fillna(0)

    # Balance
    df["LogBalance"] = np.log1p(df["Balance"])
    df["TransactionsToBalance"] = df["SumTransactions"] / df["Balance"]
    df["BalancePerProduct"] = df["Balance"] / (df["TotalProducts"] + 1)

    # Rates: how attractive is the offer compared with what the customer already has?
    df["AverageCompetitiveRate"] = rates.mean(axis=1)
    df["MaxCompetitiveRate"] = rates.max(axis=1)
    df["MinCompetitiveRate"] = rates.min(axis=1)
    df["StdCompetitiveRate"] = rates.std(axis=1)
    df["BestOfferGain"] = df["MaxCompetitiveRate"] - df["RateBefore"]
    df["AverageOfferGain"] = df["AverageCompetitiveRate"] - df["RateBefore"]
    df["ReferenceMinusBefore"] = df["ReferenceRate"] - df["RateBefore"]
    df["BestOfferVsReference"] = df["MaxCompetitiveRate"] - df["ReferenceRate"]
    df["OffersAboveRateBefore"] = rates.gt(df["RateBefore"], axis=0).sum(axis=1)

    # Activity (a count with a long tail, so log it instead of one-hot encoding it)
    df["LogActivity"] = np.log1p(df["ActivityIndicator"])
    df["ActivityPerInteraction"] = df["ActivityIndicator"] / (df["RegularInteractionIndicator"] + 1)
    df["ActivityPerProduct"] = df["ActivityIndicator"] / (df["TotalProducts"] + 1)

    # Heavy-tailed raw columns
    for column in TRANSACTION_COLUMNS + ["SumTransactions", "MaxTransaction"]:
        df[f"Log{column}"] = signed_log1p(df[column])

    return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])


# Column groups clustered with K-Means. They follow the groups explored in the
# original notebook, computed on log-scaled values so a few huge transactions do not
# dominate the distance.
DEFAULT_CLUSTER_GROUPS = {
    "ClusterBT": (["LogBalance", "LogSumTransactions"], 7),
    "ClusterRates": (["RateBefore", "AverageCompetitiveRate", "ReferenceRate"], 6),
    "ClusterActivity": (["TotalProducts", "LogSumTransactions", "LogActivity"], 7),
}


class ClusterFeatures(BaseEstimator, TransformerMixin):
    """
    Adds K-Means cluster ids and distances to the cluster centres.
    Fitted on the training data only, then applied to new data.
    """

    def __init__(self, groups: Optional[dict] = None, random_state: int = 42, add_distances: bool = True):
        self.groups = groups
        self.random_state = random_state
        self.add_distances = add_distances

    def fit(self, X: pd.DataFrame, y=None):
        self.groups_ = self.groups if self.groups is not None else DEFAULT_CLUSTER_GROUPS
        self.models_ = {}
        for name, (columns, n_clusters) in self.groups_.items():
            scaler = StandardScaler().fit(X[columns])
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.random_state)
            kmeans.fit(scaler.transform(X[columns]))
            self.models_[name] = (columns, scaler, kmeans)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for name, (columns, scaler, kmeans) in self.models_.items():
            scaled = scaler.transform(df[columns])
            df[name] = kmeans.predict(scaled)
            if self.add_distances:
                df[f"{name}Distance"] = kmeans.transform(scaled).min(axis=1)
        return df


def build_features(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Split the raw data into model inputs
    :param df_train: raw train dataframe
    :param df_test: raw test dataframe
    :return: X, y, X_test, test ids
    """
    y = df_train[TARGET_COLUMN].astype(int)
    X = add_row_features(df_train.drop(columns=[TARGET_COLUMN]))
    X_test = add_row_features(df_test)
    X_test = X_test[X.columns]
    return X, y, X_test, df_test[ID_COLUMN]


def cluster_columns(groups: Optional[dict] = None) -> List[str]:
    """
    Names of the categorical cluster-id columns
    :param groups: cluster groups
    :return: column names
    """
    return list((groups or DEFAULT_CLUSTER_GROUPS).keys())
