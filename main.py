import numpy as np

import pandas as pd

from clean_data import load_raw_data, clean_raw_data
from helpers import count_missing
from imputers import LinearModelImputer, MeanImputer
from transformers import DateTransformer, FuncTransformer, YeoJohnsonTransformer

from sklearn.model_selection import train_test_split


def split_data(X, target, group_col, test_size=.2, random_state=None):
    group_train, group_test = train_test_split(X[group_col].unique(),
                                               test_size=test_size, random_state=random_state)

    X_train, X_test = X[X[group_col].isin(group_train)], X[X[group_col].isin(group_test)]
    X_train, y_train = X_train.drop([target], axis=1).copy(), X_train[target].copy()
    X_test, y_test = X_test.drop([target], axis=1).copy(), X_test[target].copy()

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    raw_data = load_raw_data('data/training_sample.csv')
    clean_data = clean_raw_data(raw_data)

    raw_data_test = load_raw_data('data/test_sample.csv')
    clean_data_test = clean_raw_data(raw_data_test)

    imputers = [
        LinearModelImputer('units', 'rentarea', ['units <= 200', '1000 <= rentarea <= 200000']),
        LinearModelImputer('most_recent_ncf', 'recent_noi'),
        LinearModelImputer('balact_prior', 'original_loan_balance'),
        LinearModelImputer('total_value_property', 'pure_appraisal'),
        LinearModelImputer('total_value_property', 'securappvalue'),
        LinearModelImputer('origination_loan_to_value', 'securltv'),
        MeanImputer('recent_fiscal_occupied_rentspace'),
        MeanImputer('percentage_occupied_rentspace'),
        MeanImputer('current_loan_to_value_indexed'),
        MeanImputer('most_recent_fiscal_debt_service'),
        MeanImputer('debt_yield_p1'),
        MeanImputer('recent_ncf_ratio_debtservice'),
        MeanImputer('pure_appraisal_growth'),
    ]

    imputed_data = clean_data.copy()
    imputed_data_test = clean_data_test.copy()
    for imputer in imputers:
        imputed_data = imputer.fit_transform(imputed_data)
        imputed_data_test = imputer.transform(imputed_data_test)

    missing_before = count_missing(clean_data).rename('Before').to_frame()
    missing_after = count_missing(imputed_data).rename('After').to_frame()

    print('\n\nMissing Values:')
    print(missing_before.merge(missing_after, left_index=True, right_index=True).sort_values('Before', ascending=False))
    print('\n\n')

    log_transformer = lambda c: np.log(c + np.abs(np.min(c)) + 1)
    transformers = [
        DateTransformer('observation_date'),
        FuncTransformer('balact_prior', log_transformer),
        FuncTransformer('original_loan_balance', log_transformer),
        FuncTransformer('outstanding_scheduled_balance', log_transformer),
        FuncTransformer('pure_appraisal', log_transformer),
        FuncTransformer('pure_appraisal_growth', log_transformer),
        FuncTransformer('recent_noi', log_transformer),
        FuncTransformer('rentarea', log_transformer),
        FuncTransformer('securappvalue', log_transformer),
        FuncTransformer('total_value_property', log_transformer),
        FuncTransformer('units', log_transformer)
        # YeoJohnsonTransformer('balact_prior'),
        # YeoJohnsonTransformer('original_loan_balance'),
        # YeoJohnsonTransformer('outstanding_scheduled_balance'),
        # YeoJohnsonTransformer('pure_appraisal'),
        # YeoJohnsonTransformer('pure_appraisal_growth'),
        # YeoJohnsonTransformer('recent_noi'),
        # YeoJohnsonTransformer('rentarea'),
        # YeoJohnsonTransformer('securappvalue'),
        # YeoJohnsonTransformer('total_value_property'),
        # YeoJohnsonTransformer('units')
    ]

    transformed_data = imputed_data.copy()
    transformed_data_test = imputed_data_test.copy()
    for transformer in transformers:
        transformed_data = transformer.fit_transform(transformed_data)
        transformed_data_test = transformer.transform(transformed_data_test)

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # for c in transformed_data:
    #     print('Plotting %s' % c)
    #     plt.figure()
    #     sns.kdeplot(transformed_data[c])
    #     plt.title(c)
    #     plt.savefig('./figs/transformed_data_%s.pdf' % c, bbox_inches='tight')

    dummies = pd.get_dummies(
        pd.concat([transformed_data.assign(__train=1), transformed_data_test.assign(__train=0)])
    )
    final_data = dummies[dummies['__train'] == 1].drop(['__train'], axis=1).copy()
    final_data_test = dummies[dummies['__train'] == 0].drop(['__train'], axis=1).copy()

    # final_data = pd.get_dummies(transformed_data)
    # final_data_clean = pd.get_dummies(transformed_data_clean)

    X_train, X_valid, y_train, y_valid = split_data(final_data, 'target', 'masterloanidtrepp', test_size=.2, random_state=123)

    X_train = X_train.drop(['masterloanidtrepp'], axis=1)
    X_valid = X_valid.drop(['masterloanidtrepp'], axis=1)

    X_test = final_data_test.drop(['masterloanidtrepp', 'target'], axis=1)
    y_test = final_data_test['target']

    print('%s training samples and %s test samples for %s columns!' % (X_train.shape[0], X_valid.shape[0], X_train.shape[1]))
    print(final_data.dtypes)

    import xgboost
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    def train_valid_test_model(model, **fit_params):#, X_train, y_train, X_valid, y_valid, X_test, y_test):
        model.fit(X_train, y_train, **fit_params)
        pred_valid = model.predict_proba(X_valid)[:, 1]
        pred_test = model.predict_proba(X_test)[:, 1]

        fpr_valid, tpr_valid, _ = roc_curve(y_valid, pred_valid)
        fpr_test, tpr_test, _ = roc_curve(y_test, pred_test)

        auc_valid = auc(fpr_valid, tpr_valid)
        auc_test = auc(fpr_test, tpr_test)

        return fpr_valid, tpr_valid, auc_valid, fpr_test, tpr_test, auc_test

    lr_model = LogisticRegression()
    # lr_model.fit(X_train, y_train)
    # pred_lr = lr_model.predict_proba(X_valid)[:, 1]
    # fpr_lr, tpr_lr, _ = roc_curve(y_valid, pred_lr)
    # auc_lr = auc(fpr_lr, tpr_lr)
    fpr_lr, tpr_lr, auc_lr, fpr_lr_t, tpr_lr_t, auc_lr_t = train_valid_test_model(lr_model)

    xg_model = xgboost.XGBClassifier(nthread=4)#, colsample_bytree=.9, learning_rate=2, max_depth=3, n_estimators=75, subsample=1)
    # xg_model.fit(X_train, y_train, eval_metric='auc')
    # pred_xg = xg_model.predict_proba(X_valid)[:, 1]
    # fpr_xg, tpr_xg, _ = roc_curve(y_valid, pred_xg)
    # auc_xgb = auc(fpr_xg, tpr_xg)
    fpr_xg, tpr_xg, auc_xgb, fpr_xg_t, tpr_xg_t, auc_xg_t = train_valid_test_model(xg_model, eval_metric='auc')

    xg_model2 = xgboost.XGBClassifier(nthread=4, colsample_bytree=.9, learning_rate=0.1, max_depth=12, n_estimators=125, subsample=.9)
    xg_model2.fit(X_train, y_train, eval_metric='auc')
    pred_xg2 = xg_model2.predict_proba(X_valid)[:, 1]
    fpr_xg2, tpr_xg2, _ = roc_curve(y_valid, pred_xg2)
    auc_xgb2 = auc(fpr_xg2, tpr_xg2)

    import numpy as np

    np.random.seed(10)

    import matplotlib.pyplot as plt

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                                  GradientBoostingClassifier)
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc
    from sklearn.pipeline import make_pipeline

    n_estimator = 10

    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)

    # Unsupervised transformation based on totally random trees
    rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
                              random_state=0)

    rt_lm = LogisticRegression()
    pipeline = make_pipeline(rt, rt_lm)
    pipeline.fit(X_train, y_train)
    y_pred_rt = pipeline.predict_proba(X_valid)[:, 1]
    fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_valid, y_pred_rt)
    auc_rt_lm = auc(fpr_rt_lm, tpr_rt_lm)

    # Supervised transformation based on random forests
    rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
    rf_enc = OneHotEncoder()
    rf_lm = LogisticRegression()
    rf.fit(X_train, y_train)
    rf_enc.fit(rf.apply(X_train))
    rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

    y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_valid)))[:, 1]
    fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_valid, y_pred_rf_lm)
    auc_rf_lm = auc(fpr_rf_lm, tpr_rf_lm)

    grd = GradientBoostingClassifier(n_estimators=n_estimator)
    grd_enc = OneHotEncoder()
    grd_lm = LogisticRegression()
    grd.fit(X_train, y_train)
    grd_enc.fit(grd.apply(X_train)[:, :, 0])
    grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

    y_pred_grd_lm = grd_lm.predict_proba(
        grd_enc.transform(grd.apply(X_valid)[:, :, 0]))[:, 1]
    fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_valid, y_pred_grd_lm)
    auc_grd_lm = auc(fpr_grd_lm, tpr_grd_lm)

    # The gradient boosted model by itself
    y_pred_grd = grd.predict_proba(X_valid)[:, 1]
    fpr_grd, tpr_grd, _ = roc_curve(y_valid, y_pred_grd)
    auc_grd = auc(fpr_grd, tpr_grd)

    # The random forest model by itself
    y_pred_rf = rf.predict_proba(X_valid)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_valid, y_pred_rf)
    auc_rf = auc(fpr_rf, tpr_rf)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR %s' % auc_rt_lm)
    plt.plot(fpr_rf, tpr_rf, label='RF %s' % auc_rf)
    plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR %s' % auc_rf_lm)
    plt.plot(fpr_grd, tpr_grd, label='GBT %s' % auc_grd)
    plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR %s' % auc_grd_lm)
    plt.plot(fpr_lr, tpr_lr, label='LR %s' % auc_lr)
    plt.plot(fpr_xg, tpr_xg, label='XGB %s' % auc_xgb)
    plt.plot(fpr_xg_t, tpr_xg_t, label='XGB TEST %s' % auc_xg_t)
    plt.plot(fpr_xg2, tpr_xg2, label='XGB2 %s' % auc_xgb2)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show(block=False)
    #
    # plt.figure(2)
    # plt.xlim(0, 0.2)
    # plt.ylim(0.8, 1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
    # plt.plot(fpr_rf, tpr_rf, label='RF')
    # plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
    # plt.plot(fpr_grd, tpr_grd, label='GBT')
    # plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve (zoomed in at top left)')
    # plt.legend(loc='best')
    # plt.show()
    #
    # from sklearn.grid_search import GridSearchCV
    #
    # clf = xgboost.XGBClassifier(
    #     nthread=4,
    #     learning_rate=.1,
    #     max_depth=12,
    #     subsample=0.5,
    #     colsample_bytree=1.0,
    #     silent=1
    # )
    # parameters = {
    #     'learning_rate': [0.05, 0.1, 1.5, 2],
    #     'n_estimators': [75, 100, 125],
    #     'max_depth': [3, 6, 9, 12],
    #     'subsample': [0.9, 1.0],
    #     'colsample_bytree': [0.9, 1.0],
    # }
    #
    # clf = GridSearchCV(clf, parameters, n_jobs=2, cv=2, verbose=1, fit_params={'eval_metric': 'auc'})
    #
    # clf.fit(X_train, y_train)
    # best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    # print(score)
    # for param_name in sorted(best_parameters.keys()):
    #     print("%s: %r" % (param_name, best_parameters[param_name]))

    plt.show()