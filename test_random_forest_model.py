import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

from clean_data import load_raw_data, clean_raw_data
from helpers import count_missing
from imputers import LinearModelImputer, MeanImputer
from main import split_data
from transformers import DateTransformer


if __name__ == '__main__':
    raw_data = load_raw_data('data/training_sample.csv')
    clean_data = clean_raw_data(raw_data)

    imputers = [
        LinearModelImputer('units', 'rentarea', ['units <= 200', '1000 <= rentarea <= 200000']),
        LinearModelImputer('most_recent_ncf', 'recent_noi'),
        LinearModelImputer('balact_prior', 'original_loan_balance'),
        LinearModelImputer('total_value_property', 'pure_appraisal'),
        LinearModelImputer('total_value_property', 'securappvalue'),
        LinearModelImputer('origination_loan_to_value', 'securltv'),
        # MeanImputer('recent_fiscal_occupied_rentspace'),
        # MeanImputer('percentage_occupied_rentspace'),
        # MeanImputer('current_loan_to_value_indexed'),
        # MeanImputer('most_recent_fiscal_debt_service'),
        # MeanImputer('debt_yield_p1'),
        # MeanImputer('recent_ncf_ratio_debtservice'),
        # MeanImputer('pure_appraisal_growth'),
    ]

    imputed_data = clean_data.copy()
    for imputer in imputers:
        imputed_data = imputer.fit_transform(imputed_data)

    imputed_data = imputed_data.fillna(-9999999999999999);

    missing_before = count_missing(clean_data).rename('Before').to_frame()
    missing_after = count_missing(imputed_data).rename('After').to_frame()

    print('\n\nMissing Values:')
    print(missing_before.merge(missing_after, left_index=True, right_index=True).sort_values('Before', ascending=False))
    print('\n\n')

    log_transformer = lambda c: np.log(c + np.abs(np.min(c)) + 1)
    transformers = [
        DateTransformer('observation_date')
    ]

    transformers = [
        DateTransformer('observation_date')
    ]

    transformed_data = imputed_data.copy()
    for transformer in transformers:
        transformed_data = transformer.fit_transform(transformed_data)
    final_data = pd.get_dummies(transformed_data)

    X_train, X_test, y_train, y_test = split_data(final_data, 'target', 'masterloanidtrepp', test_size=.2, random_state=123)

    X_train = X_train.drop(['masterloanidtrepp'], axis=1)
    X_test = X_test.drop(['masterloanidtrepp'], axis=1)

    rf = RandomForestClassifier(max_depth=3)
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
    auc_rf_lm = auc(fpr_rf, tpr_rf)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf, label='RF %s' % auc_rf_lm)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show(block=False)

    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d %s (%f)" % (f + 1, indices[f], X_train.columns[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure(2)
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show(block=False)

    print('Done')