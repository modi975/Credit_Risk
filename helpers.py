def count_missing(data, sort=True):
    missings = data.isnull().sum() / data.shape[0]
    if sort:
        missings = missings.sort_values(ascending=False)
    return missings

def split_data(X, target, group_col, test_size=.2, random_state=None):
    group_train, group_test = train_test_split(X[group_col].unique(),
                                               test_size=test_size, random_state=random_state)

    X_train, X_test = X[X[group_col].isin(group_train)], X[X[group_col].isin(group_test)]
    X_train, y_train = X_train.drop([target], axis=1).copy(), X_train[target].copy()
    X_test, y_test = X_test.drop([target], axis=1).copy(), X_test[target].copy()

    return X_train, X_test, y_train, y_test