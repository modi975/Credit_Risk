import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import Imputer


class LinearModelImputer(Imputer):
    def __init__(self, column_x, column_y, queries=[], fit_intercept=False, normalize=False):
        self.column_x = column_x
        self.column_y = column_y
        self.queries = queries
        self.fit_intercept = fit_intercept
        self.normalize = normalize

        self.data = None
        self.lm = None
        self.mean_column_x = np.nan
        self.mean_column_y = np.nan
        self.r2 = np.nan

    def fit(self, X):
        self.data = (
            X[[self.column_x, self.column_y]]
            .drop_duplicates()
            .dropna()
        )

        for query in self.queries:
            self.data = self.data.query(query)

        self.mean_column_x = self.data[self.column_x].mean()
        self.mean_column_y = self.data[self.column_y].mean()
        self.lm = LinearRegression(fit_intercept=self.fit_intercept, normalize=self.normalize)
        self.lm.fit(self.data[[self.column_x]], self.data[self.column_y])
        self.r2 = r2_score(self.data[self.column_y], self.lm.predict(self.data[[self.column_x]]))

        print('Fit LinearModelImputer between %s and %s on %s observations, R2 = %s' % (
            self.column_x, self.data.shape[0], self.column_y, self.r2
        ))

    def transform(self, X):
        X = X.copy()

        col_x_isnull = X[self.column_x].isnull()
        col_y_isnull = X[self.column_y].isnull()

        got_none = X[col_x_isnull & col_y_isnull]
        got_x = X[(~ col_x_isnull) & col_y_isnull]
        got_y = X[col_x_isnull & (~ col_y_isnull)]

        X.loc[got_none.index, self.column_x] = self.mean_column_x
        X.loc[got_none.index, self.column_y] = self.mean_column_y

        if got_x.shape[0] > 0:
            X.loc[got_x.index, self.column_y] = self.lm.predict(got_x[[self.column_x]])

        if got_y.shape[0] > 0:
            intercept = self.lm.coef_[0] if self.fit_intercept else 0
            X.loc[got_y.index, self.column_x] = (got_y[self.column_y] - intercept) / self.lm.coef_[-1]

        print('LinearModelImputer filled in %s observations for %s and %s' % (
            got_x.shape[0] + got_y.shape[0], self.column_x, self.column_y
        ))

        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class FunctionImputer(Imputer):
    def __init__(self, column, func):
        self.column = column
        self.func = func
        self.result = np.nan

    def fit(self, X):
        self.result = self.func(X[self.column].dropna())

    def transform(self, X):
        X = X.copy()

        index_missing = X[X[self.column].isnull()].index
        X.loc[index_missing, self.column] = self.result

        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class MeanImputer(FunctionImputer):
    def __init__(self, column):
        self.column = column
        self.func = np.mean


class MedianImputer(FunctionImputer):
    def __init__(self, column):
        self.column = column
        self.func = np.median


class ValueImputer(FunctionImputer):
    def __init__(self, column, value):
        self.column = column
        def f(*args, **kwargs):
            return value
        self.func = lambda: f
