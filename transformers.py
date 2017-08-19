from __future__ import division

from sklearn.base import TransformerMixin

from YeoJohnson import YeoJohnson


class DateTransformer(TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit_transform(self, X, y=None, **fit_params):
        def transform_func(date_string):
            xs = list(map(int, date_string.split('_')))
            year, month = xs[0], xs[1]

            return year + (month - 1) / 12

        X = X.copy()
        X[self.column] = X[self.column].map(transform_func)

        return X

    def transform(self, X):
        return self.fit_transform(X)


class FuncTransformer(TransformerMixin):
    def __init__(self, column, func):
        self.column = column
        self.func = func

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        X = X.copy()
        X[self.column] = self.func(X[self.column])

        return X


class YeoJohnsonTransformer(TransformerMixin):
    def __init__(self, column):
        self.column = column

        self.yj = YeoJohnson()

    def fit_transform(self, X, y=None, **fit_params):
        X = X.copy()
        X[self.column] = self.yj.fit(X[self.column], 2)

        return X

    def transform(self, X):
        raise NotImplementedError()

