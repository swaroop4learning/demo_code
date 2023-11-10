import os
import yaml
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score
from sklearn.metrics import precision_recall_curve, confusion_matrix
from data.model_dao import get_db_con, get_validation_data


class PerformanceFeatures():

    def __init__(self, model):
        self.model = model
        print(os.environ['complai_home'])
        self.model_home = os.environ['complai_home']+"/"+model
        print('self.model_home', self.model_home)
        yml_path = self.model_home+"/"+model+"_config.yml"
        with open(yml_path, 'r') as yaml_file:
            self.config = yaml.safe_load(yaml_file)
        db_path = self.model_home + "/data/db.json"
        self.db = get_db_con(db_path)
        self.periods = {0: 'Day1', 1: 'Day2', 2: 'Day3', 3: 'Day4', 4: 'Day5', 5: 'Day6', 6: 'Day7'}
        self.problem_type = self.config['problem_type']
        self.target_classes = self.config['target_classes']
        #print('config is ', self.config)

    def load_datasets_old(self):
        '''
           Function to Load Training and Validation Datasets. This function will not be required when data will be provided with config file.
           Args:
               None
           Returns:
               Train Dataset, Validation Dataset, Validation Labels, Complete Validation Dataset with labels
           '''
        val_path = self.config['validation_raw_path']
        val_df = pd.read_pickle(self.model_home+val_path)
        target_label_column = self.config['target_label_column']
        target_pred_label_column = self.config['target_pred_label_column']
        target_pred_proba_column = self.config['target_pred_proba_column']
        val_label_df = val_df[target_label_column]
        val_pred_df = val_df[target_pred_label_column]
        val_pred_proba_df = val_df[target_pred_proba_column]
        self.target_cols_drop_lt = [target_label_column, target_pred_label_column, target_pred_proba_column]
        val_data = val_df.drop(columns=self.target_cols_drop_lt)
        return val_df, val_label_df, val_pred_df, val_pred_proba_df, val_data

    def load_datasets(self):
        validation_data = get_validation_data(self.db)
        val_df = pd.DataFrame(validation_data).dropna()
        val_label_df = val_df['ground_truth']
        if(self.problem_type=='regression'):
            cols_to_drop = ['nearest_synthetic_counterfactual_sparsity', 'distance_values', 'avg_robustness',
                            'explainability_score', 'feature_imporntance_cf', 'num_features_changes','latest_record_ind']
            val_df = val_df.drop(cols_to_drop, axis=1)
            val_pred_df = val_df['regression_value']
            val_pred_proba_df = val_df['regression_value']
            self.target_cols_drop_lt = ['ground_truth', 'regression_value']
        elif(self.problem_type=='multiclass'):
            cols_to_drop = ['nearest_synthetic_counterfactual_sparsity', 'distance_values', 'avg_robustness_0',
                            'avg_robustness_1', 'avg_robustness_2', 'min_robustness', 'min_class_label',
                            'explainability_score', 'num_features_changes', 'feature_imporntance_cf',
                            'latest_record_ind']
            val_df = val_df.drop(cols_to_drop, axis=1)
            val_pred_df = val_df['prediction_label']
            val_pred_proba_df = val_df['pred_proba_scores']
            self.target_cols_drop_lt = ['ground_truth', 'prediction_label', 'pred_proba_scores']
        else:
            cols_to_drop = ['nearest_synthetic_counterfactual_sparsity', 'distance_values', 'avg_robustness_0',
                            'avg_robustness_1', 'min_robustness', 'min_class',
                            'explainability_score', 'num_features_changes', 'feature_imporntance_cf',
                            'latest_record_ind']
            val_df = val_df.drop(cols_to_drop, axis=1)
            val_pred_df = val_df['prediction_label']
            val_pred_proba_df = val_df['probability_score']
            self.target_cols_drop_lt = ['ground_truth', 'prediction_label', 'probability_score']
        val_data = val_df.drop(columns=self.target_cols_drop_lt)
        return val_df, val_label_df, val_pred_df, val_pred_proba_df, val_data


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

    def custom_predict(self, pred_proba_df, threshold):
        '''
        Function to output custom prediction class labels with respect to defined
        threshold value (probability greater than which it is positive class)
        Args:
            model (sklearn model): Model for Which Prediction will be called
            x_data (pandas datafrane): Dataset on which predict will be called
            threshold (float): Probability threshold below which class is 0 and above which class is 1
        Returns:
            predicted labels (numpy array): Predicted Label array with class labels
        '''
        y_pred_lab = []
        probs = pred_proba_df
        if(self.problem_type=='binary-classification'):
            for i in probs:
                if i < threshold:
                    y_pred_lab.append(int(0))
                else:
                    y_pred_lab.append(int(1))
        else:
            y_pred_lab = pred_proba_df
        return np.array(y_pred_lab, dtype=int)

    def model_performance_report(self, y_val, y_pred_lab):
        '''
        Show Model Performance Report
        Args:
            model (sklearn model): Pretrained Model
            x_val (pandas dataframe): Validation Dataset
            y_val (numpy array): True Labels
            y_pred_lab (numpy array): Predicted Labels
        Returns:
            Displays Streamit Metric Row consisting of All Model performance metric
        '''
        if(self.problem_type=='binary-classification'):
            pr = precision_score(y_val, y_pred_lab)
            rc = recall_score(y_val, y_pred_lab)
            f1 = f1_score(y_val, y_pred_lab)
        else:
            pr = precision_score(y_val, y_pred_lab, average='weighted')
            rc = recall_score(y_val, y_pred_lab, average='weighted')
            f1 = f1_score(y_val, y_pred_lab, average='weighted')
        ac = accuracy_score(y_val, y_pred_lab)
        d = {}
        d['Accuracy'] = ac
        d['Misclass Rate'] = 1 - ac
        d['F1 Score'] = f1
        d['Precision'] = pr
        d['Recall'] = rc
        # New Visualization
        for i in list(d.keys()):
            d[i] = d[i] * 100
            d[i] = "{:.2f}".format(d[i])
        return {"Accuracy": d['Accuracy'], "Misclass": d['Misclass Rate'], "F1 Score": d['F1 Score'],
                    "Precision": d['Precision'], "Recall": d['Recall']}

    def regression_performance_report(self, y_val, y_pred):
        '''
        Show Model Performance Report
        Args:
            model (sklearn model): Pretrained Model
            x_val (pandas dataframe): Validation Dataset
            y_val (numpy array): True Labels
            y_pred_lab (numpy array): Predicted Labels
        Returns:
            Displays Streamit Metric Row consisting of All Model performance metric
        '''
        #features = x_val.columns.to_list()
        features = self.get_feature_names()
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred, squared=True)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        m_log_e = mean_squared_log_error(y_val, y_pred)
        med_abs_er = median_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        adjusted_r2 = 1 - (1 - r2 ** 2) * ((len(y_val) - 1) / (len(y_val) - len(features) - 1))
        d = {}
        d['Mean Absolute Error'] = mae
        d['Mean Squared Error'] = mse
        d['Root Mean Squared Error'] = rmse
        d['Mean Squared Log Error'] = m_log_e
        d['Median Absolute Error'] = med_abs_er
        d['R Squared'] = r2
        d['Adjusted R Sqaured'] = adjusted_r2

        # New Visualization
        for i in list(d.keys()):
            d[i] = "{:.2f}".format(d[i])

        return{"Mean Absolute Error": d['Mean Absolute Error'], "Mean Squared Error": d['Mean Squared Error'],
                    "Root Mean Squared Error": d['Root Mean Squared Error'],
                    "Mean Squared Log Error": d['Mean Squared Log Error'],
                    "Median Absolute Error": d['Median Absolute Error'], "R2 Sqaured": d['R Squared'],
                    'Adjusted R Sqaured': d['Adjusted R Sqaured']}

    def show_confusion_matrix(self, y_val, y_pred_lab):
        '''
        Plot Confusion Matrix
        Args:
            y_val (numpy array): True Labels
            y_pred_lab (numpy array): Predicted Label
            target_names (list of string): Final Outcomes (e.g ['Non-Fraud', 'Fraud'])
            cmap (string): Colormap for confusion matrix (e.g 'Blue','Inferno')
            Normalize (bool): Whether to normalize (show percentage only)
        '''
        #target_names = self.config['target_class_definitions']
        target_names = list(self.target_classes.keys())
        cm = confusion_matrix(y_val, y_pred_lab)
        return cm, target_names

    def create_slices(self, dataset, chosen_feature_1=None, chosen_feature_2=None, chosen_bucket_1=None,
                      chosen_bucket_2=None):
        '''
        Given Feature Names and Buckets, this fucntion will generate slices from the dataset (with labels appended)
        and return list of dataframes and correspodning list of true labels. Returns (None,None,None) otherwise.
        Args:
            dataset (pandas Dataframe): Dataframe on which slices to be created
            chosen_feature_1 (string): First Feature Name
            chosen_feature_2 (string): Second Feature Name
            chosen_bucket_1 (int): Bucket Number for First Feature (Keep None if First Feature is Categorical)
            chosen_bucket_2 (int): Bucket Number for Second Feature (Keep None if Second Feature is Categorical)
        '''

        slice_list = []
        slice_label_list = []
        slice_pred_proba_list = []
        report_dict = {}
        feature_value_ranges_1 = []
        feature_value_ranges_2 = []
        num_datapoints = []
        target_feature = 'ground_truth'
        if(self.problem_type == 'binary-classification'):
            target_pred_proba_column = 'probability_score'
        elif(self.problem_type == 'regression'):
            target_pred_proba_column = 'regression_value'
        else:
            target_pred_proba_column = 'prediction_label'
        feature_names = self.get_feature_names()
        cat_feat = self.config['categorical_features_indexes']

        # If two chosen features are same
        if(None in cat_feat):
            numerical_features = feature_names
            categorical_features = []
        else:
            categorical_features = [dataset.columns[i] for i in cat_feat]  # Fix: Will be provided via config file
            numerical_features = [i for i in feature_names if i not in categorical_features]
        if (chosen_feature_1 == chosen_feature_2):
            chosen_feature_1 = chosen_feature_2
            chosen_feature_2 = None
            st.write('Primary and Secondary Features Cannot be Same')
            return None, None, None, None
        # If first feature is categorical and second feature is categorical
        if ((chosen_feature_1 in categorical_features) and (chosen_feature_2 in categorical_features)):
            cats_1 = dataset[chosen_feature_1].unique()
            cats_2 = dataset[chosen_feature_2].unique()

            for i in range(len(cats_1)):
                for j in range(len(cats_2)):
                    sliced_df = dataset[
                        (dataset[chosen_feature_1] == cats_1[i]) & (dataset[chosen_feature_2] == cats_2[j])]
                    if len(sliced_df) > 0:
                        feature_value_ranges_1.append(cats_1[i])
                        feature_value_ranges_2.append(cats_2[j])
                        num_datapoints.append(len(sliced_df))
                        sliced_data = sliced_df.drop(columns=self.target_cols_drop_lt)
                        sliced_label = sliced_df[target_feature]
                        sliced_label_pred_prob = sliced_df[target_pred_proba_column]
                        slice_list.append(sliced_data)
                        slice_label_list.append(sliced_label)
                        slice_pred_proba_list.append(sliced_label_pred_prob)
            report_dict['1st Feature Value Ranges'] = feature_value_ranges_1
            report_dict['2nd Feature Value Ranges'] = feature_value_ranges_2
            report_dict['Slice Size'] = num_datapoints
            return slice_list, slice_label_list, slice_pred_proba_list, report_dict
        # If first feature is numerical and second feature is categorical
        elif ((chosen_feature_1 in numerical_features) and (chosen_feature_2 in categorical_features)):
            if chosen_bucket_1 == None:
                st.write('Choose Number of Buckets for Primary Feature')
                return None, None, None
            else:

                feature_1_min = dataset[chosen_feature_1].min()
                feature_1_max = dataset[chosen_feature_1].max()
                cats_2 = dataset[chosen_feature_2].unique()
                feature_1_width = (feature_1_max - feature_1_min) / chosen_bucket_1
                start = feature_1_min
                end = 0
                feature_1_tuples = []
                for i in range(chosen_bucket_1):
                    if i == 0:
                        start = feature_1_min
                        end = feature_1_min + (i + 1) * feature_1_width
                        feature_1_tuples.append((start, end))
                    elif i < (chosen_bucket_1 - 1):
                        start = end
                        end = feature_1_min + (i + 1) * feature_1_width
                        feature_1_tuples.append((start, end))
                    else:
                        start = end
                        end = feature_1_max
                        feature_1_tuples.append((start, end))
                if isinstance(feature_1_max, int):
                    feature_1_tuples = list(tuple(map(int, tup)) for tup in feature_1_tuples)
                for i in range(len(feature_1_tuples)):
                    for j in range(len(cats_2)):
                        sliced_df = dataset[(dataset[chosen_feature_1] >= feature_1_tuples[i][0]) & (
                                dataset[chosen_feature_1] < feature_1_tuples[i][1]) & (
                                                    dataset[chosen_feature_2] == cats_2[j])]
                        if len(sliced_df) > 0:
                            feature_value_ranges_1.append(feature_1_tuples[i])
                            feature_value_ranges_2.append(cats_2[j])
                            num_datapoints.append(len(sliced_df))
                            sliced_data = sliced_df.drop(columns=self.target_cols_drop_lt)
                            sliced_label = sliced_df[target_feature]
                            sliced_label_pred_prob = sliced_df[target_pred_proba_column]
                            slice_list.append(sliced_data)
                            slice_label_list.append(sliced_label)
                            slice_pred_proba_list.append(sliced_label_pred_prob)
                report_dict['1st Feature Value Ranges'] = feature_value_ranges_1
                report_dict['2nd Feature Value Ranges'] = feature_value_ranges_2
                report_dict['Slice Size'] = num_datapoints
                return slice_list, slice_label_list, slice_pred_proba_list, report_dict
        # If first feature is categorical and second feature is numerical
        elif ((chosen_feature_1 in categorical_features) and (chosen_feature_2 in numerical_features)):
            if chosen_bucket_2 == None:
                st.write('Choose Number of Buckets for Primary Feature')
                return None, None, None
            else:

                cats_1 = dataset[chosen_feature_1].unique()
                feature_2_min = dataset[chosen_feature_2].min()
                feature_2_max = dataset[chosen_feature_2].max()
                feature_2_width = (feature_2_max - feature_2_min) / chosen_bucket_2
                start = feature_2_min
                end = 0
                feature_2_tuples = []
                for i in range(chosen_bucket_2):
                    if i == 0:
                        start = feature_2_min
                        end = feature_2_min + (i + 1) * feature_2_width
                        feature_2_tuples.append((start, end))
                    elif i < (chosen_bucket_2 - 1):
                        start = end
                        end = feature_2_min + (i + 1) * feature_2_width
                        feature_2_tuples.append((start, end))
                    else:
                        start = end
                        end = feature_2_max
                        feature_2_tuples.append((start, end))
                if isinstance(feature_2_max, int):
                    feature_2_tuples = list(tuple(map(int, tup)) for tup in feature_2_tuples)
                for i in range(len(feature_2_tuples)):
                    for j in range(len(cats_1)):
                        sliced_df = dataset[(dataset[chosen_feature_2] >= feature_2_tuples[i][0]) & (
                                dataset[chosen_feature_2] < feature_2_tuples[i][1]) & (
                                                    dataset[chosen_feature_1] == cats_1[j])]
                        if len(sliced_df) > 0:
                            feature_value_ranges_1.append(cats_1[j])
                            feature_value_ranges_2.append(feature_2_tuples[i])
                            num_datapoints.append(len(sliced_df))
                            sliced_data = sliced_df.drop(columns=self.target_cols_drop_lt)
                            sliced_label = sliced_df[target_feature]
                            sliced_label_pred_prob = sliced_df[target_pred_proba_column]
                            slice_list.append(sliced_data)
                            slice_label_list.append(sliced_label)
                            slice_pred_proba_list.append(sliced_label_pred_prob)
                report_dict['1st Feature Value Ranges'] = feature_value_ranges_1
                report_dict['2nd Feature Value Ranges'] = feature_value_ranges_2
                report_dict['Slice Size'] = num_datapoints
                return slice_list, slice_label_list, slice_pred_proba_list, report_dict
        # If first feature is numerical and second feature is numerical also
        elif ((chosen_feature_1 in numerical_features) and (chosen_feature_2 in numerical_features)):
            if (chosen_bucket_1 == None):
                st.write('Choose Number of Buckets for Primary Feature')
                return None, None
            elif ((chosen_bucket_1 != None) and (chosen_bucket_2 == None)):
                st.write('Choose Number of Buckets for Secondary Feature')
                return None, None, None
            else:

                feature_1_min = round(dataset[chosen_feature_1].min(),2)
                feature_1_max = round(dataset[chosen_feature_1].max(),2)
                feature_2_min = round(dataset[chosen_feature_2].min(),2)
                feature_2_max = round(dataset[chosen_feature_2].max(),2)
                feature_1_width = round((feature_1_max - feature_1_min) / chosen_bucket_1,2)
                start = feature_1_min
                end = 0
                feature_1_tuples = []
                for i in range(chosen_bucket_1):
                    if i == 0:
                        start = feature_1_min
                        end = feature_1_min + (i + 1) * feature_1_width
                        feature_1_tuples.append((start, end))
                    elif i < (chosen_bucket_1 - 1):
                        start = end
                        end = feature_1_min + (i + 1) * feature_1_width
                        feature_1_tuples.append((start, end))
                    else:
                        start = end
                        end = feature_1_max
                        feature_1_tuples.append((start, end))
                feature_2_width = round((feature_2_max - feature_2_min) / chosen_bucket_2,2)
                start = feature_2_min
                end = 0
                feature_2_tuples = []
                for i in range(chosen_bucket_2):
                    if i == 0:
                        start = feature_2_min
                        end = feature_2_min + (i + 1) * feature_2_width
                        feature_2_tuples.append((start, end))
                    elif i < (chosen_bucket_2 - 1):
                        start = end
                        end = feature_2_min + (i + 1) * feature_2_width
                        feature_2_tuples.append((start, end))
                    else:
                        start = end
                        end = feature_2_max
                        feature_2_tuples.append((start, end))
                if isinstance(feature_1_max, int):
                    feature_1_tuples = list(tuple(map(int, tup)) for tup in feature_1_tuples)
                if isinstance(feature_2_max, int):
                    feature_2_tuples = list(tuple(map(int, tup)) for tup in feature_2_tuples)
                for i in range(len(feature_1_tuples)):
                    for j in range(len(feature_2_tuples)):
                        sliced_df = dataset[(dataset[chosen_feature_1] >= feature_1_tuples[i][0]) & (
                                dataset[chosen_feature_1] < feature_1_tuples[i][1]) & (
                                                    dataset[chosen_feature_2] >= feature_2_tuples[j][0]) & (
                                                    dataset[chosen_feature_2] < feature_2_tuples[j][1])]
                        if len(sliced_df) > 0:
                            feature_value_ranges_1.append(feature_1_tuples[i])
                            feature_value_ranges_2.append(feature_2_tuples[j])
                            num_datapoints.append(len(sliced_df))
                            sliced_data = sliced_df.drop(columns=self.target_cols_drop_lt)
                            sliced_label = sliced_df[target_feature]
                            sliced_label_pred_prob = sliced_df[target_pred_proba_column]
                            slice_list.append(sliced_data)
                            slice_label_list.append(sliced_label)
                            slice_pred_proba_list.append(sliced_label_pred_prob)
                #print('the feature value range1 is ',feature_value_ranges_1)
                #print('the feature value range2 is ', feature_value_ranges_2)
                report_dict['1st Feature Value Ranges'] = feature_value_ranges_1
                report_dict['2nd Feature Value Ranges'] = feature_value_ranges_2
                report_dict['Slice Size'] = num_datapoints
                return slice_list, slice_label_list, slice_pred_proba_list, report_dict

    def predict_slices(self, slices, labels, pred_scores, report_dict, threshold):
        '''
        Given List of Slices where each slice is a dataset, generate model report for each
        slice with custom threshold. Expected Return Dataframe with Report Details.
        Args:
            slices (list of pandas dataframe): List of all the slices
            labels (list of pandas dataframe): List of all the labels corresponding to the slices
            model (sklearn model): Model on which performance to be measured
            report_dict (dictionary): Dictionary of slice information (such as Slice size, value ranges) obtained from function call
                                      create_slices(dataset,chosen_feature_1 = None, chosen_feature_2 = None, chosen_bucket_1 = None, chosen_bucket_2 = None)
            threshold (float): Probability Cutoff / Threshold on which performance is to be checked
        Returns:
            report_df (pandas dataframe): Dataframe containing performance report and slice information for each slice
        '''
        acc_list = []
        pr_list = []
        rec_list = []
        f1_list = []
        for i in range(len(slices)):
            x_data = slices[i]
            y_val = labels[i]
            #y_pred_lab = pred_scores[i]
            y_pred_lab = self.custom_predict(pred_scores[i], threshold)
            if (self.problem_type == 'binary-classification'):
                pr = precision_score(y_val, y_pred_lab)
                rc = recall_score(y_val, y_pred_lab)
                f1 = f1_score(y_val, y_pred_lab)
            else:
                pr = precision_score(y_val, y_pred_lab, average='weighted')
                rc = recall_score(y_val, y_pred_lab, average='weighted')
                f1 = f1_score(y_val, y_pred_lab, average='weighted')
            ac = accuracy_score(y_val, y_pred_lab)
            ac = "{:.2f}".format(ac)
            pr = "{:.2f}".format(pr)
            rc = "{:.2f}".format(rc)
            f1 = "{:.2f}".format(f1)
            acc_list.append(ac)
            pr_list.append(pr)
            rec_list.append(rc)
            f1_list.append(f1)
        report_dict['Accuracy'] = acc_list
        report_dict['F1 Score'] = f1_list
        report_dict['Precision'] = pr_list
        report_dict['Recall'] = rec_list
        report_df = pd.DataFrame(report_dict, index=[i for i in range(len(report_dict['Slice Size']))])
        return report_df

    def predict_slices_regression(self, slices, labels, pred_scores, report_dict, threshold=None):
        '''
        Given List of Slices where each slice is a dataset, generate model report for each
        slice with custom threshold. Expected Return Dataframe with Report Details.
        Args:
            slices (list of pandas dataframe): List of all the slices
            labels (list of pandas dataframe): List of all the labels corresponding to the slices
            model (sklearn model): Model on which performance to be measured
            report_dict (dictionary): Dictionary of slice information (such as Slice size, value ranges) obtained from function call
                                      create_slices(dataset,chosen_feature_1 = None, chosen_feature_2 = None, chosen_bucket_1 = None, chosen_bucket_2 = None)
        Returns:
            report_df (pandas dataframe): Dataframe containing performance report and slice information for each slice
        '''
        mae_list = []
        mse_list = []
        rmse_list = []
        m_log_e_list = []
        med_abs_er_list = []
        r2_list = []
        for i in range(len(slices)):
            x_data = slices[i]
            y_val = labels[i]
            # y_pred_lab = custom_predict(model, x_data, threshold)
            y_pred_lab = self.custom_predict(pred_scores[i], threshold)
            mae = mean_absolute_error(y_val, y_pred_lab)
            mse = mean_squared_error(y_val, y_pred_lab, squared=True)
            rmse = mean_squared_error(y_val, y_pred_lab, squared=False)
            m_log_e = mean_squared_log_error(y_val, y_pred_lab)
            med_abs_er = median_absolute_error(y_val, y_pred_lab)
            r2 = r2_score(y_val, y_pred_lab)
            mae = "{:.2f}".format(mae)
            mse = "{:.2f}".format(mse)
            rmse = "{:.2f}".format(rmse)
            m_log_e = "{:.2f}".format(m_log_e)
            med_abs_er = "{:.2f}".format(med_abs_er)
            r2 = "{:.2f}".format(r2)
            mae_list.append(mae)
            mse_list.append(mse)
            rmse_list.append(rmse)
            m_log_e_list.append(m_log_e)
            med_abs_er_list.append(med_abs_er)
            r2_list.append(r2)
        report_dict['Mean Absolute Error'] = mae_list
        report_dict['Mean Squared Error'] = mse_list
        report_dict['Root Mean Squared Error'] = rmse_list
        report_dict['Mean Squared Log Error'] = m_log_e_list
        report_dict['Median Absolute Error'] = med_abs_er_list
        report_dict['R2 Score'] = r2_list
        report_df = pd.DataFrame(report_dict, index=[i for i in range(len(report_dict['Slice Size']))])
        return report_df


