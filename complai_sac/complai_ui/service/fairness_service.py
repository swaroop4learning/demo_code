import os
import yaml
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from data.model_dao import get_db_con, get_fairness_data, get_validation_data


class FairnessService():

    def __init__(self, model):
        self.model = model
        self.model_home = os.environ['complai_home'] + "/" + model
        yml_path = self.model_home + "/" + model + "_config.yml"
        with open(yml_path, 'r') as yaml_file:
            self.config = yaml.safe_load(yaml_file)
        db_path = self.model_home + "/data/db.json"
        self.db = get_db_con(db_path)
        # self.periods = {0: 'Day1', 1: 'Day2', 2: 'Day3', 3: 'Day4', 4: 'Day5', 5: 'Day6', 6: 'Day7'}
        self.problem_type = self.config['problem_type']
        self.target_classes = self.config['target_classes']
        if self.target_classes!=None:
            self.avg_robustness = ["avg_robustness_"+i for i in self.target_classes]
        self.prefered_class = self.config['prefered_class']
        self.feature_combinations = self.config['protected_feature_combinations']
        if (self.problem_type == 'binary'):
            self.probability_threshold = self.config['probability_threshold']
        self.protected_features = self.config['protected_attributes']
        #print('protected features at initialization time are ', self.protected_features)
        if(None not in self.protected_features):
            self.protected_features_mappings = self.config['protected_attributes_mappings']
        # print('config is ', self.config)

    #This method needs to be added to utils
    def load_datasets(self):
        validation_data = get_validation_data(self.db)
        val_df = pd.DataFrame(validation_data).dropna()
        val_label_df = val_df['ground_truth']
        if (self.problem_type == 'regression'):
            cols_to_drop = ['nearest_synthetic_counterfactual_sparsity', 'distance_values', 'avg_robustness',
                            'explainability_score', 'feature_importance_cf', 'num_features_changes',
                            'latest_record_ind']
            val_df = val_df.drop(cols_to_drop, axis=1)
            val_pred_df = val_df['regression_value']
            val_pred_proba_df = val_df['regression_value']
            self.target_cols_drop_lt = ['ground_truth', 'regression_value']
        elif (self.problem_type == 'multiclass'):
            cols_to_drop = ['nearest_synthetic_counterfactual_sparsity', 'distance_values', 'min_robustness',
                            'min_class_label',
                            'explainability_score', 'num_features_changes', 'feature_importance_cf',
                            'latest_record_ind']
            cols_to_drop.extend(self.avg_robustness)
            val_df = val_df.drop(cols_to_drop, axis=1)
            val_pred_df = val_df['prediction_label']
            val_pred_proba_df = val_df['pred_proba_scores']
            self.target_cols_drop_lt = ['ground_truth', 'prediction_label', 'pred_proba_scores']
        else:
            cols_to_drop = ['nearest_synthetic_counterfactual_sparsity', 'distance_values', 'min_robustness',
                            'min_class',
                            'explainability_score', 'num_features_changes', 'feature_importance_cf',
                            'latest_record_ind']
            cols_to_drop.extend(self.avg_robustness)
            val_df = val_df.drop(cols_to_drop, axis=1)
            val_pred_df = val_df['prediction_label']
            val_pred_proba_df = val_df['probability_score']
            self.target_cols_drop_lt = ['ground_truth', 'prediction_label', 'probability_score']
        val_data = val_df.drop(columns=self.target_cols_drop_lt)
        return val_df, val_label_df, val_pred_df, val_pred_proba_df, val_data

    def get_protected_data(self, protected_attrs, data_df, combinations):
        for i in combinations:
            comb = i + "_" + combinations[i]
            data_df[comb] = data_df[i].astype(str) + "_" + data_df[combinations[i]].astype(str)
            protected_attrs.append(comb)
        return protected_attrs, data_df

    def get_di_scores(self, data_df, protected_column, threshold_value):
        di_scores = {}
        data_df['new_pred_target'] = data_df['probability_score'].apply(lambda x: 0 if x<threshold_value else 1)
        for j in data_df[protected_column].unique():
            tot_records = len(data_df[data_df[protected_column] == j])
            pos_records = len(data_df[(data_df[protected_column] == j) & (data_df['new_pred_target'] == 1)])
            di_score = pos_records / tot_records
            di_scores[j] = di_score
        return di_scores

    def get_eo_scores(self, data_df, protected_column, threshold_value):
        eo_scores = {}
        data_df['new_pred_target'] = data_df['probability_score'].apply(lambda x: 0 if x < threshold_value else 1)
        for j in data_df[protected_column].unique():
            tot_records = len(data_df[data_df[protected_column] == j])
            pos_records = len(data_df[(data_df[protected_column] == j) & (data_df['new_pred_target'] == 1) & (data_df['ground_truth'] == 1)])
            eo_score = pos_records / tot_records
            eo_scores[j] = eo_score
        return eo_scores

    def get_legal_scores(self, data_df, protected_column, threshold_value, legal_metric):
        if(legal_metric=='Equal Opportunity'):
            scores = self.get_eo_scores(data_df, protected_column, threshold_value)
        elif(legal_metric=='Disparate Impact'):
            scores = self.get_di_scores(data_df, protected_column, threshold_value)
        overall_min = min(scores.values())
        overall_max = max(scores.values())
        overall_score = (overall_min / overall_max) * 100
        return scores, overall_score



    def get_decrypted_data(self, data_df, protected_features_mappings):
        for i in protected_features_mappings:
            org_dict = protected_features_mappings[i]
            reversed_dictionary = {int(value): key for (key, value) in org_dict.items()}
            data_df = data_df.replace({i: reversed_dictionary})
        return data_df

    def get_conf_metrics(self, target, target_pred):
        conf_metrics = {}
        tn, fp, fn, tp = confusion_matrix(target, target_pred).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        fdr = fp / (fp + tp)
        tnr = tn / (tn + fp)
        fnr = fn / (fn + tp)
        conf_metrics['Recall'] = tpr
        conf_metrics['False Positive Rate'] = fpr
        conf_metrics['False Discovery Rate'] = fdr
        conf_metrics['True Negative Rate'] = tnr
        conf_metrics['False Negative Rate'] = fnr
        return conf_metrics

    def get_metrics(self, target, target_pred):
        metrics = {}
        metrics['Precision'] = precision_score(target, target_pred)
        metrics['f1_score'] = f1_score(target, target_pred)
        metrics.update(self.get_conf_metrics(target, target_pred))
        return metrics

    def get_model_fairness(self, data_df, protected_column, threshold_value):
        metrics = {}
        data_df['new_pred_target'] = data_df['probability_score'].apply(lambda x: 0 if x < threshold_value else 1)
        for j in data_df[protected_column].unique():
            records_df = data_df[data_df[protected_column] == j]
            targets = records_df['ground_truth']
            targets_pred = records_df['new_pred_target']
            metrics[j] = self.get_metrics(targets, targets_pred)
        return metrics

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

    # def get_fairness_data_mock(self):
    #     fairness_data = {
    #         "fairness_info": {
    #             "Sex": {
    #                 "1": 0.6938202247191011,
    #                 "0": 0.5566037735849056
    #             }
    #         },
    #         "di_info": {
    #             "Sex": {
    #                 "1": 0.4887640449438202,
    #                 "0": 0.20754716981132076
    #             }
    #         },
    #         "latest_record_ind": "Y"
    #     }
    #     return fairness_data

    def get_feature_fairness(self, protected_feature, fairness_metric):
        fairness_data = get_fairness_data(self.db)[0]
        #print('fairness data is ', fairness_data)
        if(fairness_metric=='Synthetic Flip Test'):
            fairness_feature_data = fairness_data['fairness_info'][protected_feature]
        else:
            fairness_feature_data = fairness_data['di_info'][protected_feature]

        feature_mapping = self.protected_features_mappings[protected_feature]
        #protected_feature_attribution = dict((feature_mapping[key], value) for (key, value) in fairness_feature_data.items())
        protected_feature_attribution = {}
        feature_attributions = []
        #print('fairness is ',fairness_feature_data)
        for i,v in feature_mapping.items():
            protected_feature_attribution[i] = fairness_feature_data[str(v)]*100
            feature_attributions.append(fairness_feature_data[str(v)]*100)
        if(fairness_metric=='Synthetic Flip Test'):
            overall_min_protected = min(feature_attributions)
        else:
            min_val = min(feature_attributions)
            max_val = max(feature_attributions)
            overall_min_protected = round((min_val/max_val)*100,2)
        return protected_feature_attribution, overall_min_protected

    def get_chi_square_test(self, df, column1, column2):
        #print(df.info())
        status = 0
        df_cnt = pd.DataFrame(df[column1])
        df_cnt["Frequency"] = 0.0
        df_cnt = df_cnt.groupby(column1).agg({"Frequency": "count"}).reset_index()
        df_cnt_obs = pd.DataFrame(df[column2])
        df_cnt_obs["Frequency"] = 0.0
        df_cnt_obs = df_cnt_obs.groupby(column2).agg({"Frequency": "count"}).reset_index()
        df_chi_square = pd.merge(df_cnt, df_cnt_obs, how="left", left_on=column1,
                                 right_on=column2)
        df_chi_square.fillna(value={"Frequency_x": 0.0, "Frequency_y": 0.0, column1: "other", column2: "other"},
                             inplace=True)
        # df_chi_square.fillna("other", inplace=True)
        chisq_result = stats.chisquare(f_obs=df_chi_square["Frequency_y"], f_exp=df_chi_square["Frequency_x"])
        if chisq_result[1]<0.05:
            status = 1
        return status

    def CorrelationMatrix_x(self, pr_X, pr_features, numerical_features, categorical_features):
        '''
        :param pr_X: test dataset containing all features
        Calculates Anova and chi square the associated p-value.
        :return:
        '''
        #print('data_info:', pr_X)
        corrMatrix = pd.DataFrame()
        cols = list(pr_X.columns)
        for i in pr_features:
            #print('i is ',i)
            cols.remove(i)
        #print('cols: ', cols)
        # corrMatrix.index = pr_features
        i = 0
        for prr in pr_features:
            for att in cols:
                #print(pr_X[att].dtypes)
                if pr_X[att].dtypes != 'object':
                    # if (att in numerical_features) & (pr_X[att].dtypes != 'object'):
                    CategoryGroupLists = pr_X.groupby(prr)[att].apply(list)
                    corrMatrix.at[i, 'protected_attribute'] = prr
                    corrMatrix.at[i, 'attribute'] = att
                    anova_value = round(stats.f_oneway(*CategoryGroupLists)[1],2)  # P-Value for Anova
                    status=0
                    if anova_value < 0.05:
                        status = 1
                    corrMatrix.at[i, 'value'] =  status
                    i += 1
                else:
                    temp_data = pr_X[[prr, att]]
                    temp_data[att] = temp_data[att].astype('str')
                    #print(temp_data.head())
                    corrMatrix.at[i, 'protected_attribute'] = prr
                    corrMatrix.at[i, 'attribute'] = att
                    corrMatrix.at[i, 'value'] = round(self.get_chi_square_test(temp_data, prr, att),2)  # P-Value for chi2
        #print(corrMatrix.T)

        df = corrMatrix[['protected_attribute', 'attribute', 'value']]
        corrMatrix_x = df.pivot_table(index='protected_attribute', columns='attribute', values='value')
        corrMatrix_x.replace(0.0, 'Not Detected', inplace=True)
        corrMatrix_x.replace(1.0, 'Proxy Detected', inplace=True)
        return corrMatrix_x


    def get_correlation_matrix_df(self, protected_features):
        cat_feat = self.config['categorical_features_indexes']
        validation_data = get_validation_data(self.db)
        val_df = pd.DataFrame(validation_data).dropna()
        features = self.get_feature_names()
        val_df = val_df[features]
        categorical_features = [val_df.columns[i] for i in cat_feat]
        numerical_features = [i for i in features if i not in categorical_features]
        protected_features = protected_features
        #print('the protected attributes from corrl are ',protected_features)
        correlation_matrix = self.CorrelationMatrix_x(val_df, protected_features, numerical_features, categorical_features)
        #print('the correlation matrix is ', correlation_matrix)
        return correlation_matrix




