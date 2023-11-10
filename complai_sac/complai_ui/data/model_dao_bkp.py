import streamlit as st
import pandas as pd
import numpy as np
from tinydb import TinyDB, Query

@st.cache
def get_train_data():
    train_df = pd.DataFrame(np.random.randn(2000, 3), columns=['feature1', 'feature2', 'feature3'])
    return train_df

@st.cache(allow_output_mutation=True)
def get_db_con(db_path):
    try:
        db_con = TinyDB(db_path)
        return db_con
    except Exception as e:
        print('Could not establish connection with db ', e)
        raise

@st.cache
def get_live_predictions_data_old(db_con):
    try:
        live_predictions = db_con.table('live_predictions')
        live_predictions_data = live_predictions.search(Query().latest_record_ind == "Y")
        return live_predictions_data
    except Exception as e:
        print('Could not load data for live_predictions from db connection ', e)
        raise


@st.cache
def get_multi_live_predictions_data(db_con):
    try:
        live_predictions = db_con.table('live_predictions')
        live_predictions_data = live_predictions.all()
        return live_predictions_data
    except Exception as e:
        print('Could not load data for live_predictions from db connection ', e)
        raise


@st.cache
def get_validation_data(db_con):
    try:
        val_predictions = db_con.table('predictions_values_counterfactuals')
        val_predictions_data = val_predictions.search(Query().latest_record_ind == "Y")
        return val_predictions_data
    except Exception as e:
        print('Could not load data for validation data predictions from db connection ', e)
        raise


@st.cache
def get_shap_aggr_data(db_con, data_type, target=None):
    try:
        if(data_type == 'val'):
            shap_val_aggr = db_con.table('shap_values_validation_aggregated')
            if(target is None):
                shap_val_aggr_data = shap_val_aggr.search(Query().latest_record_ind == "Y")
            else:
                shap_val_aggr_data = shap_val_aggr.search(Query().latest_record_ind == "Y" and Query().target == target)
            return shap_val_aggr_data
        else:
            shap_live_aggr = db_con.table('shap_values_live_aggregated')
            shap_live_aggr_data = shap_live_aggr.search(Query().latest_record_ind == "Y")
            return shap_live_aggr_data
    except Exception as e:
        print('Could not load data for validation data predictions from db connection ', e)
        raise

@st.cache
def get_ndcg_data(db_con):
    try:
        ndcg_aggr = db_con.table('shap_aggregated_values')
        ndcg_aggr_data = ndcg_aggr.search(Query().latest_record_ind == "Y")
        return ndcg_aggr_data
    except Exception as e:
        print('Could not load ndgc data from db connection ', e)
        raise

@st.cache
def get_fairness_data(db_con):
    try:
        fairness = db_con.table('fairness_values')
        fairness_data = fairness.search(Query().latest_record_ind == "Y")
        return fairness_data
    except Exception as e:
        print('Could not load fairness data from db connection ', e)
        raise

@st.cache
def get_performance_data(db_con):
    try:
        performance = db_con.table('performance_values')
        performance_data = performance.all()
        return performance_data
    except Exception as e:
        print('Could not load ndgc data from db connection ', e)
        raise