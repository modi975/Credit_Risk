import pandas as pd


column_rename_dict = {
    'bad_flag_final_v3': 'target',
    'appvalue': 'pure_appraisal',
    'fmrappvalue': 'total_value_property',
    'gr_appvalue': 'pure_appraisal_growth',
    'obal': 'outstanding_scheduled_balance',
    'origloanbal': 'original_loan_balance',
    'mrfytdocc': 'percentage_occupied_rentspace',
    'oltv': 'origination_loan_to_value',
    'oterm': 'original_maturity',
    'priorfyncf': 'most_recent_ncf',
    'priorfydscrncf': 'recent_ncf_ratio_debtservice',
    'priorfynoi': 'recent_noi',
    'priorfyocc': 'recent_fiscal_occupied_rentspace',
    'priorfydscr': 'most_recent_fiscal_debt_service',
    'cltv_1': 'current_loan_to_value_indexed'
}

dropped_columns = ['appvalue_prior', 'mrappvalue', 'gr_mrappvalue', 'sample', 'changeinvalue', 'balact', 'gr_balact',
                   'msa','observation_date']


def load_raw_data(path):
    return pd.read_csv(path, index_col=0)


def clean_raw_data(raw_data):
    return raw_data.rename(columns=column_rename_dict).drop(dropped_columns, axis=1)
