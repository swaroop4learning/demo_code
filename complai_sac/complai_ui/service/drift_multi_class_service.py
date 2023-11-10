
import os
import math
import yaml
import pandas as pd
import numpy as np
from scipy.spatial import distance
from data.model_dao import get_db_con, get_validation_data, get_shap_aggr_data, get_ndcg_data, get_multi_live_predictions_data, get_drift_aggr_data

class DriftMultiService():

    def __init__(self, model):
        self.model = model
        self.model_home = os.environ['complai_home'] + "/" + model
        yml_path = self.model_home + "/" + model + "_config.yml"
        with open(yml_path, 'r') as yaml_file:
            self.config = yaml.safe_load(yaml_file)
        db_path = self.model_home+"/data/db.json"
        self.db = get_db_con(db_path)
        self.periods = {0: 'Day1', 1: 'Day2', 2: 'Day3', 3: 'Day4', 4: 'Day5', 5: 'Day6', 6: 'Day7'}
        self.problem_type = self.config['problem_type']
        self.target_classes = self.config['target_classes']
        # print('config is ', self.config)

    def get_feature_names(self):
        '''
        Returns Feature Names (Supplied from Config). Currently hardcoded but will not be required when provided as part of config file
        Args:
            None
        Returns:
            all_feature_names (list of strings): List of Features of the loaded dataset
        '''
        all_feature_names = self.config['feature_names']
        return all_feature_names


    def get_prediction_labels(self, live_predictions_dict, threshold):
        df1 = pd.DataFrame(live_predictions_dict['data'])
        df1 = df1.loc[df1['latest_record_ind']=='Y']
        df = df1.loc[:, df1.columns != 'latest_record_ind'].dropna()
        df['prediction_label'] = df['pred_proba_scores'].apply(lambda x : np.argmax(x))
        values_to_return = df['prediction_label'].values
        return values_to_return

    def get_live_pred_probas(self, live_predictions_dict, target_class):
        df1 = pd.DataFrame(live_predictions_dict['data'])
        df1 = df1.loc[df1['latest_record_ind'] == 'Y']
        df = df1.loc[:, df1.columns != 'latest_record_ind'].dropna()
        df['pred_scores'] = df['pred_proba_scores'].apply(lambda x : x[target_class])
        df_arr = df['pred_scores'].values
        df_arr = df_arr.astype('float64')
        return df_arr

    def get_live_label_distribution(self, threshold):
        '''
        Returns live label distribution for live dataset with custom threshold
        Args:
            model (sklearn model): Model for Which Live Label Distribution is to be observed
            threshold (float): Probability threshold below which class is 0 and above which class is 1
        Returns:
            label_dist_dict (dictionary): Keys are Periods and values are tuples of  (% of negative labels, % of positive labels)
        '''
        #periods = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        unique_labels = list(self.target_classes.values())
        actual_unique_labels = list(self.target_classes.keys())
        data_dict = dict(periods=list(self.periods.values()))
        for i in actual_unique_labels:
            data_dict[i] = []
        live_predictions_data = get_multi_live_predictions_data(self.db)
        label_dist_dict = {}
        for i,v in self.periods.items():
            data = live_predictions_data[i]
            labels = self.get_prediction_labels(data, threshold)
            for actual_class in actual_unique_labels:
                data_dict[actual_class].append(np.count_nonzero(labels == self.target_classes[actual_class])/len(labels))
            #label_dist_dict[v] = tuple((np.count_nonzero(labels == i)/len(labels)) for i in unique_labels)
        return data_dict


    def get_live_prediction_drift(self, target_class_label):
        '''
        Returns live prediction drift distribution for live dataset.
        Args:
            model (sklearn model): Model for Which Live Label Distribution is to be observed
            baseline_data (pandas dataframe): Validation dataset (typically from the same distribution as training dataset)
        Returns:
            report_dist_dict (dictionary): Key as Days and values as tuple of (JS Score, avg fraud pred, number of datapoints)
        '''
        #periods = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        cols_to_drop = ['nearest_synthetic_counterfactual_sparsity', 'distance_values', 'avg_robustness_0',
                        'avg_robustness_1', 'avg_robustness_2', 'min_robustness', 'min_class_label',
                        'explainability_score', 'num_features_changes', 'feature_importance_cf', 'latest_record_ind']
        report_dist_dict = {}
        target_class = self.target_classes[target_class_label]
        live_predictions_data = get_multi_live_predictions_data(self.db)
        val_data = get_validation_data(self.db)
        val_df = pd.DataFrame(val_data).dropna()
        #val_df = val_df_data.drop(cols_to_drop, axis=1)
        val_df['pred_scores'] = val_df['pred_proba_scores'].apply(lambda x: x[target_class])
        baseline_preds = val_df['probability_score'].values
        for i, v in self.periods.items():
            # Jensen Shannon Divergence
            data = live_predictions_data[i]
            live_preds = self.get_live_pred_probas(data, target_class)
            avg_fraud_pred = np.mean(live_preds)
            distance_val = 0

            # Point of Concern

            if len(live_preds) == len(baseline_preds):
                distance_val = distance.jensenshannon(baseline_preds, live_preds)
            elif len(live_preds) > len(baseline_preds):
                live_preds_sample = np.random.choice(live_preds, size=len(baseline_preds))
                distance_val = distance.jensenshannon(baseline_preds, live_preds_sample)
            elif len(live_preds) < len(baseline_preds):
                baseline_preds_sample = np.random.choice(baseline_preds, size=len(live_preds))
                distance_val = distance.jensenshannon(baseline_preds_sample, live_preds)
            report_dist_dict[v] = (distance_val, avg_fraud_pred, len(live_preds))
        return report_dist_dict

    def NDCG_Score(self, reference_attribute_dict, live_attribute_dict):
        '''
        Function to calculate the similarity in feature-ranking between the two scenarios (baseline vs live)
        Args:
            reference_attribute_dict (dictionary): Feature Attribute Dictionary for Baseline (validation dataset) obtained using get_feature_attribution(train_dataset, dataset, model, method)
            live_attribute_dict (dictionary): Feature Attribute Dictionary for live dataset obtained using get_feature_attribution(train_dataset, dataset, model, method)
        Returns:
            NDCG (float): Normalized Discounted Cumulative Gain Score for measuring similarity of Baseline Feature Ranking and Live Feature ranking
        '''
        sorted_reference_attribute = dict(
            sorted(reference_attribute_dict.items(), key=lambda item: item[1], reverse=True))
        sorted_reference_attribute_keys = list(sorted_reference_attribute.keys())
        sorted_live_attribute = dict(sorted(live_attribute_dict.items(), key=lambda item: item[1], reverse=True))
        sorted_live_attribute_keys = list(sorted_live_attribute.keys())
        iDCG = 0
        for i in range(1, len(sorted_reference_attribute) + 1):
            iDCG += sorted_reference_attribute[sorted_reference_attribute_keys[i - 1]] / math.log(i + 1, 2)
        DCG = 0
        for i in range(1, len(sorted_live_attribute_keys) + 1):
            DCG += sorted_reference_attribute[sorted_live_attribute_keys[i - 1]] / math.log(i + 1, 2)
        ndcg = DCG / iDCG
        return ndcg*100

    def get_ndcg_class(self, target_class):
        ndcg_aggr_data = get_ndcg_data(self.db)
        #print('the taget class is ', target_class)
        ndcg_class_score = {i['target']:i['ndcg_score'] for i in ndcg_aggr_data}[target_class]
        return ndcg_class_score*100

    def get_ndcg_scores(self):
        ndcg_aggr_data = get_ndcg_data(self.db)
        #ndcg_score = ndcg_aggr_data[day]['Drift_score_day']
        ndcg_scores = [i['ndcg_score'] for i in ndcg_aggr_data]
        overall_ndcg_score = min(ndcg_scores)
        return overall_ndcg_score*100


    def get_baseline_feature_attribution(self, target):
        baseline_data = get_drift_aggr_data(self.db, 'train', target)
        baseline_feature_attribution = {i['Column_Name']: i['train_attribution_value'] for i in baseline_data}
        return baseline_feature_attribution

    def get_val_data_feature_attribution(self, target):
        #live_data = get_shap_aggr_data(self.db, data_type='live')
        val_data = get_drift_aggr_data(self.db, data_type='val')
        #live_shap_data = live_data[i]['live_shap_df_aggregated']
        val_feature_attribution = {i['validation_feature_attribution_df_aggregated']['Column_Name']: i['validation_feature_attribution_df_aggregated']['validation_attribution_value'] for i in val_data}
        #print('val_feature_attribution is ', val_feature_attribution)
        return val_feature_attribution

    def get_live_data_feature_attribution(self,i, target):
        live_data = get_shap_aggr_data(self.db, data_type='live')
        live_shap_data = live_data[i]['live_shap_df_aggregated']
        live_shap_data_df = pd.DataFrame(live_shap_data)
        live_shap_data_df = live_shap_data_df.loc[live_shap_data_df['target'] == target]
        live_feature_attribution = dict(zip(live_shap_data_df.Column_Name, live_shap_data_df.live_shap_aggregated))
        return live_feature_attribution


    def get_number_of_features_drifted(self, target_class):
        '''
        Returns dictionary of daily number of features drifted for live data
        Args:
            model (sklearn model): Pretrained Model
            train_dataset (pandas dataset): Training Dataset
            baseline_feature_attribution (dictionary): Feature Attribution Dictionary obtained from get_feature_attribution(train_dataset, dataset, model, method) for baseline dataset
            method (string): Method in which feature attribution is to be calculated for live data (typically same as method used for calculation of baseline_feature_attribution)
        Returns:
            drift_count_dict (dictionary): Keys are periods and values are names of features that are drifted on that day
        '''
        #periods = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        baseline_feature_attribution = self.get_baseline_feature_attribution(target_class)
        sorted_feature_attribution_dict = dict(
            sorted(baseline_feature_attribution.items(), key=lambda item: item[1], reverse=True))
        #live_data = get_shap_aggr_data(self.db, data_type='live')
        drift_count_dict = {}
        for i, v in self.periods.items():
            live_feature_attribution = self.get_live_data_feature_attribution(i, target_class)
            sorted_live_feature_attribution = dict(
                sorted(live_feature_attribution.items(), key=lambda item: item[1], reverse=True))
            drifted_features = []
            names_base = list(sorted_feature_attribution_dict.keys())
            names_drifted = list(sorted_live_feature_attribution.keys())
            for j in range(len(names_drifted)):
                if names_drifted[j] != names_base[j]:
                    drifted_features.append(names_drifted[j])
            drift_count_dict[v] = drifted_features
        return drift_count_dict

    def get_summary_feature_attribution_distributions(self, feature_name, target_class):
        '''
        Returns dictionary of daily feature attribution of live data for a specific feature
        Args:
            feature_name (string): The feature for which summarized feature attributions for entire week to be obtained
        Returns:
            drift_value_dict (dictionary): Keys are periods and values are feature attributes of live data for the particular period
        '''
        drift_value_dict = {}
        for i, v in self.periods.items():
            live_feature_attribution = self.get_live_data_feature_attribution(i, target_class)
            drift_value_dict[v] = live_feature_attribution[feature_name]
        return drift_value_dict


