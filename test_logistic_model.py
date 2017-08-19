import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from clean_data import load_raw_data, clean_raw_data
from helpers import count_missing
from imputers import LinearModelImputer
from main import split_data
from transformers import DateTransformer, FuncTransformer, YeoJohnsonTransformer

if __name__ == '__main__':
    raw_data = load_raw_data('data/training_sample.csv')
    clean_data = clean_raw_data(raw_data)

    imputers = [
        LinearModelImputer('balact_prior', 'original_loan_balance'),
        LinearModelImputer('total_value_property', 'pure_appraisal'),
        LinearModelImputer('units', 'rentarea', ['units <= 200', '1000 <= rentarea <= 200000']),
        LinearModelImputer('most_recent_ncf', 'recent_noi'),
    ]

    imputed_data = clean_data.copy()
    for imputer in imputers:
        imputed_data = imputer.fit_transform(imputed_data)

    # imputed_data = imputed_data.fillna(-9999999999999999)

    missing_before = count_missing(clean_data).rename('Before').to_frame()
    missing_after = count_missing(imputed_data).rename('After').to_frame()

    print('\n\nMissing Values:')
    print(missing_before.merge(missing_after, left_index=True, right_index=True).sort_values('Before', ascending=False))
    print('\n\n')

    log_transformer = lambda c: np.log(c + np.abs(np.min(c)) + 1)
    transformers = [
        DateTransformer('observation_date'),
        FuncTransformer('balact_prior', log_transformer),
        FuncTransformer('pure_appraisal', log_transformer),
        FuncTransformer('recent_noi', log_transformer),
        FuncTransformer('units', log_transformer),
        FuncTransformer('percentage_occupied_rentspace', log_transformer),
        FuncTransformer('outstanding_scheduled_balance', log_transformer)
    ]

    transformed_data = imputed_data.copy()
    for transformer in transformers:
        transformed_data = transformer.fit_transform(transformed_data)
    final_data = pd.get_dummies(transformed_data)

    final_data = final_data[['target', 'masterloanidtrepp', 'observation_date', 'pure_appraisal', 'balact_prior', 'units', 'recent_noi']].copy()

    X_train, X_test, y_train, y_test = split_data(final_data, 'target', 'masterloanidtrepp', test_size=.2, random_state=123)

    X_train = X_train.drop(['masterloanidtrepp'], axis=1)
    X_test = X_test.drop(['masterloanidtrepp'], axis=1)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict_proba(X_test)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
    auc_rf_lm = auc(fpr_lr, tpr_lr)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_lr, tpr_lr, label='LR %s' % auc_rf_lm)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show(block=True)

