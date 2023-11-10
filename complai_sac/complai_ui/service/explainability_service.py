import os
import yaml
import pandas as pd
import numpy as np
import streamlit as st
from tinydb import TinyDB, Query
#from nice.explainers import NICE
import itertools
import bokeh
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.layouts import widgetbox, Row
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import ColumnDataSource, Panel, Tabs, CDSView, GroupFilter
from data.model_dao import get_db_con, get_validation_data, get_shap_aggr_data

class ExplainabilityService():

    def __init__(self, model):
        self.model = model
        self.model_home = os.environ['complai_home'] + "/" + model
        yml_path = self.model_home + "/" + model + "_config.yml"
        with open(yml_path, 'r') as yaml_file:
            self.config = yaml.safe_load(yaml_file)
        db_path = self.model_home+"/data/db.json"
        self.db = get_db_con(db_path)
        #self.periods = {0: 'Day1', 1: 'Day2', 2: 'Day3', 3: 'Day4', 4: 'Day5', 5: 'Day6', 6: 'Day7'}
        self.problem_type = self.config['problem_type']
        self.target_classes = self.config['target_classes']
        if self.target_classes!=None:
            self.avg_robustness = ["avg_robustness_"+i for i in self.target_classes]
        #print('the target classes are ', self.target_classes)
        self.prefered_class = self.config['prefered_class']
        self.id_column = self.config['id_column']
        if(self.problem_type=='binary'):
            self.probability_threshold = self.config['probability_threshold']
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

    def get_baseline_feature_attribution(self, target_class = None):
        baseline_data = get_shap_aggr_data(self.db, 'val', target_class)
        baseline_feature_attribution = {i['Column_Name']: i['validation_shap_aggregated'] for i in baseline_data}
        sorted_feature_attribution_dict = dict(
            sorted(baseline_feature_attribution.items(), key=lambda item: item[1], reverse=True))
        return sorted_feature_attribution_dict

    def return_different_features(self, original, counterfactual, display=False):
        '''
        Shows Dataframe with Features whose value differ in original and counterfactual
        Args:
            original (pd.Dataframe): Original Instance (usually from validation dataset)
            counterfactual (pd.Dataframe): Corresponding counterfactual instance
                                           (usually obtained from either nearest_synthetic_counterfactual or nearest_counterfactual call)
            display (bool): If you want to display the generated report to streamlit then True else False
        Returns:
            explanation_df (pandas dataframe): Dataframe containing Feature Name, Original Value, Counterfact Value
        '''
        feature_names = original.columns.to_list()
        new_features = []
        orig_values = []
        cf_Values = []
        for i in range(len(feature_names)):
            if (original.iloc[0][feature_names[i]] != counterfactual.iloc[0][feature_names[i]]):
                new_features.append(feature_names[i])
                orig_values.append(original.iloc[0][feature_names[i]])
                cf_Values.append(counterfactual.iloc[0][feature_names[i]])
        explanation_dict_new = {}
        explanation_dict_new['Feature'] = new_features
        explanation_dict_new['Original Value'] = orig_values
        explanation_dict_new['Counterfactual Value'] = cf_Values
        explanation_df = pd.DataFrame(explanation_dict_new, index=[i for i in range(len(new_features))],
                                      columns=['Feature', 'Original Value', 'Counterfactual Value'])
        if (display == True):
            st.table(explanation_df.assign(index='').set_index('index'))
        return explanation_df

    def feature_importance_CF(self, display):
        validation_data = get_validation_data(self.db)
        val_df = pd.DataFrame(validation_data).dropna()
        feature_names = self.get_feature_names()
        val_df = val_df[feature_names]
        report_dict = {}
        # Initialization
        for i in feature_names:
            report_dict[i] = 0

        for i in range(len(val_df)):
            x_dict = dict(val_df.iloc[i])
            original = pd.DataFrame(x_dict, index=[0], columns=feature_names)
            counterfactual_dict = validation_data[i]['nearest_synthetic_counterfactual_sparsity']
            counterfactual_df = pd.DataFrame(counterfactual_dict, index = [0], columns = feature_names)
            explanation_df = self.return_different_features(original, counterfactual_df, display)
            for i in explanation_df['Feature']:
                report_dict[i] += 1

        for i in feature_names:
            report_dict[i] /= len(val_df)

        return report_dict


    def get_explainability_scores(self):
        val_data = get_validation_data(self.db)
        val_df = pd.DataFrame(val_data).dropna()
        val_df['num_features_changes'] = val_df['num_features_changes'].apply(lambda x: 6 if x > 5 else x)
        total_records = len(val_df)
        explainability_dict = {}
        for i in range(1,7):
            count_of_records = len(val_df.loc[val_df['num_features_changes'] == i])
            explainability_dict[i] = (count_of_records/total_records)*100
        explainability_score = val_df['explainability_score'][0]*100
        return explainability_dict, explainability_score

    def get_regression_robustness_score(self):
        val_data = get_validation_data(self.db)
        val_df = pd.DataFrame(val_data).dropna()
        avg_robustness = val_df['avg_robustness'][0]*100
        return avg_robustness

    def get_robustness_scores(self):
        target_classes = list(self.target_classes.keys())
        val_data = get_validation_data(self.db)
        val_df = pd.DataFrame(val_data).dropna()
        total_records = len(val_df)
        robustness_dict = {}
        robustness_lt = []
        for i_class in target_classes:
            i = self.target_classes[i_class]
            robustness_col = "avg_robustness_" + str(i_class)
            robustness_score = val_df[robustness_col][0] * 100
            robustness_lt.append(robustness_score)
            count_records = len(val_df.loc[val_df['prediction_label'] == i]) / total_records
            robustness_dict[i_class] = (robustness_score , count_records * 100)
        min_robustness = min(robustness_lt)
        avg_robustness = sum(robustness_lt) / len(robustness_lt)
        return avg_robustness, min_robustness, robustness_dict

    def get_real_counterfactuals(self, index):
        # target_classes = list(self.target_classes.keys())
        val_data = get_validation_data(self.db)
        df_data_real = pd.DataFrame.from_records([val_data[index]['nearest_counterfactual_l1']])
        return df_data_real

    def get_synthetic_counterfactuals(self, index):
        #target_classes = list(self.target_classes.keys())
        val_data = get_validation_data(self.db)
        df_data_syn = pd.DataFrame.from_records([val_data[index]['nearest_synthetic_counterfactual_sparsity']])
        return df_data_syn

    def get_val_data_by_index(self, val_df, index):
        features = self.get_feature_names()
        if(self.problem_type=='regression'):
            prediction_label = val_df['regression_value'].iloc[index]
            actual_prediction_label = prediction_label
            prediction_probability = actual_prediction_label
        else:
            prediction_label = val_df['prediction_label'].iloc[index]
            actual_prediction_label = [k for k, v in self.target_classes.items() if v == prediction_label][0]
            prediction_probability = val_df['probability_score'].iloc[index]
        return val_df[features].iloc[index].to_frame().T, actual_prediction_label, prediction_probability

    def return_different_features(self, original, counterfactual, display=False):
        '''
        Shows Dataframe with Features whose value differ in original and counterfactual
        Args:
            original (pd.Dataframe): Original Instance (usually from validation dataset)
            counterfactual (pd.Dataframe): Corresponding counterfactual instance
                                           (usually obtained from either nearest_synthetic_counterfactual or nearest_counterfactual call)
            display (bool): If you want to display the generated report to streamlit then True else False
        Returns:
            explanation_df (pandas dataframe): Dataframe containing Feature Name, Original Value, Counterfact Value
        '''
        feature_names = original.columns.to_list()
        new_features = []
        orig_values = []
        cf_Values = []
        for i in range(len(feature_names)):
            if (original.iloc[0][feature_names[i]] != counterfactual.iloc[0][feature_names[i]]):
                new_features.append(feature_names[i])
                orig_values.append(original.iloc[0][feature_names[i]])
                cf_Values.append(counterfactual.iloc[0][feature_names[i]])
        explanation_dict_new = {}
        explanation_dict_new['Feature'] = new_features
        explanation_dict_new['Original Value'] = orig_values
        explanation_dict_new['Counterfactual Value'] = cf_Values
        explanation_df = pd.DataFrame(explanation_dict_new, index=[i for i in range(len(new_features))],
                                      columns=['Feature', 'Original Value', 'Counterfactual Value'])
        if (display == True):
            explanation_df['Original Value'] = explanation_df['Original Value'].astype("string")
            explanation_df['Counterfactual Value'] = explanation_df['Counterfactual Value'].astype("string")
            st.table(explanation_df.assign(index='').set_index('index'))
        return explanation_df

    def get_bins_from_pred(self, y_preds, num_bins):
        '''
        Get Binned Frequency Plots from Predictions
        Args:
            y_preds (numpy array): Predictions from model.predict
            num_bins (int): Number of Bins
        Returns:
            Result_Dict (dictionary): key is bin number and value is bin count
            bin_thresholds (list of tuples): thresholds of bins
        '''
        y_min = np.min(y_preds)
        y_max = np.max(y_preds)
        y_width = (y_max - y_min) / num_bins
        start = y_min
        end = 0
        y_tuples = []
        for i in range(num_bins):
            if i == 0:
                start = y_min
                end = y_min + (i + 1) * y_width
                y_tuples.append((start, end))
            elif i < (num_bins - 1):
                start = end
                end = y_min + (i + 1) * y_width
                y_tuples.append((start, end))
            else:
                start = end
                end = y_max
                y_tuples.append((start, end))
        bin_counts = [0 for i in range(num_bins)]
        for i in y_preds:
            for j in range(len(y_tuples)):
                if (i >= y_tuples[j][0] and i < y_tuples[j][1] and j < len(y_tuples) - 1):
                    bin_counts[j] += 1
                    break
                elif (i >= y_tuples[j][0] and i <= y_tuples[j][1] and j == len(y_tuples) - 1):
                    bin_counts[j] += 1
                    break
        for i in range(len(bin_counts)):
            bin_counts[i] = (bin_counts[i] / y_preds.shape[0]) * 100
        return bin_counts, y_tuples

    def get_prefered_class_label(self):
        label_name = [k for k, v in self.target_classes.items() if v == self.prefered_class][0]
        #print('the label name is ', label_name)
        return label_name

    def plot_prediction_histogram(self, bin_counts, y_bin_thresholds):
        '''
        Prediction Plot
        Args:
            bin_counts (list): Bin Frequency Percentages
            y_bin_thresholds (list of tuples): Bin Thresholds
        Plot:
            Histogram Plot
        '''
        # Data Preparation
        bin_names_list = []
        for i in range(len(y_bin_thresholds)):
            bin_names_list.append('Bin ' + str(i + 1))

        y_new_thresholds = []
        for i in range(len(y_bin_thresholds)):
            y_new_thresholds.append(("{:.2f}".format(y_bin_thresholds[i][0]), "{:.2f}".format(y_bin_thresholds[i][1])))

        data = dict(bin_names=bin_names_list,
                    thresholds=y_new_thresholds,
                    frequencies=bin_counts)

        source = ColumnDataSource(data)

        # Tools
        hover = HoverTool(tooltips=[("Bin Name", "@bin_names"),
                                    ("Bin Threshold", "@thresholds"),
                                    ("Bin Frequency", "@frequencies")])
        TOOLS = [hover]

        # Figure
        p = figure(tools=TOOLS,
                   x_range=bin_names_list,
                   x_axis_label='Bins',
                   y_axis_label='Bin Frequency',
                   plot_height=400, plot_width=900,
                   title="Prediction Histogram Plot")
        p.vbar(x='bin_names', top='frequencies', source=source, color='#6baed6', width=0.8,
               legend_label='Bin Frequency Percentage')

        p.xgrid.grid_line_color = None
        if len(y_bin_thresholds) <= 25:
            p.xaxis.major_label_orientation = "horizontal"
        else:
            p.xaxis.major_label_orientation = "vertical"
        p.legend.location = 'top_right'
        p.legend.click_policy = "mute"
        p.y_range.start = 0
        st.bokeh_chart(p, use_container_width=True)


    def render_plot_widget(self, x_dataset, y_pred_label, y_pred_prob, target_class):
        '''
        Fucntion to Render Plot Widget
        Args:
            x_dataset (pandas dataframe): Dataset on which prediction is to be made
            y_pred_label (numpy array): Predicted Labels (Fraud:1 and Non-Fraud:0)
            y_pred_prob (numpoy array): Predicted Probability of Fraudness
        Returns:
            Displays Bokeh Plot for Probability Prediction for each model along with the Bokeh Table containing
            datapoint IDs, corresponding prediction class, prediction probability
        '''
        tabs = Tabs(tabs=[self.bokeh_prediction_plot(x_dataset, y_pred_label, y_pred_prob, target_class)])
        st.bokeh_chart(tabs)

    def bokeh_prediction_plot(self, x_dataset, y_pred_label, y_pred_prob, target_class):
        '''
        Funciton to plot interactive Prediction Plot from Database.
        Args:
            x_dataset (pandas dataframe): Dataset on which prediction is to be made
            y_pred_label (numpy array): Predicted Labels (Fraud:1 and Non-Fraud:0)
            y_pred_prob (numpoy array): Predicted Probability of Fraudness
        Returns:
            Bokeh Plot Object for Probability Prediction for each model
        '''
        # Datasource for Plot
        #neg_class_name = list(self.target_classes.keys())[0]
        #pos_class_name = list(self.target_classes.keys())[1]
        labels_df = x_dataset['ground_truth']
        neg_class_name = "Not "+str(target_class)
        pos_class_name = str(target_class)
        ID_ = [i for i in range(len(x_dataset))]
        Class_ = []
        colors = []
        classification_status = []
        for i in range(y_pred_label.shape[0]):
            if y_pred_label[i] == self.target_classes[target_class]:
                Class_.append(pos_class_name)
                colors.append('red')
            else:
                Class_.append(neg_class_name)
                colors.append('green')
            if labels_df[i] == y_pred_label[i]:
                classification_status.append("Classified")
            else:
                classification_status.append("Misclassified")
        Positive_class_Prob = list(y_pred_prob)

        # for i in range(len(Positive_class_Prob)):
        #     if (Positive_class_Prob[i] < 0.5):
        #         colors.append('green')
        #     else:
        #         colors.append('red')
        # Data for Bokeh Scatter Plot
        data = dict(id_val=ID_,
                    pred_class=Class_,
                    fraud_prob=Positive_class_Prob,
                    cl_status=classification_status,
                    color_code=colors)
        source = ColumnDataSource(data)
        columns = [TableColumn(field="id_val", title="Datapoint ID"),
                   TableColumn(field="pred_class", title="Predicted Class"),
                   TableColumn(field="fraud_prob", title="Potential Positive Class Probability"),
                   TableColumn(field="cl_status", title="Classification")]
        t = DataTable(source=source, columns=columns, width=400, height=600)
        # Scatter Plot Configuration
        hover = HoverTool(tooltips=[("Datapoint ID", "@id_val"),
                                    ("Predicted Class", "@pred_class"),
                                    ("Positive Class Probability", "@fraud_prob"),
                                    ("Classification", "@cl_status")])
        TOOLS = [hover, 'box_select', 'lasso_select', 'poly_select', 'tap', 'reset']
        # Create a view for each label
        fraud_filter = [GroupFilter(column_name='pred_class', group=pos_class_name)]
        fraud_view = CDSView(source=source, filters=fraud_filter)
        nonfraud_filter = [GroupFilter(column_name='pred_class', group=neg_class_name)]
        nonfraud_view = CDSView(source=source, filters=nonfraud_filter)
        # Figure Arguments
        common_figure_kwargs = {'tools': TOOLS, 'plot_width': 650, 'plot_height': 600, 'x_axis_label': 'Datapoint ID',
                                'y_axis_label': pos_class_name+' Probability', 'title': pos_class_name+' Probability Plot'}
        common_circle_kwargs = {'x': 'id_val', 'y': 'fraud_prob', 'source': source, 'size': 10, 'alpha': 1, }
        common_fraud_kwargs = {'view': fraud_view, 'color': 'red', 'legend_label': 'Predicted '+pos_class_name}
        common_nonfraud_kwargs = {'view': nonfraud_view, 'color': 'green', 'legend_label': 'Predicted '+neg_class_name}
        # Figure Configuration
        p = figure(**common_figure_kwargs)
        p.circle(**common_circle_kwargs, **common_fraud_kwargs, muted_alpha=0.1)
        p.circle(**common_circle_kwargs, **common_nonfraud_kwargs, muted_alpha=0.1)
        p.legend.click_policy = 'mute'
        # Passing Widget to Bokeh Tab
        row_display = bokeh.layouts.Row(children=[p, t], sizing_mode="stretch_width")
        return bokeh.models.Panel(child=row_display, title="Database Prediction Plot")

    def plot_explanation_report(self, report_dict):
        '''
        Plots the Explanation Report obtained from create_explanation_report
        Args:
            Report Dict (dictionary): Obtained from create_explanation_report
            method (string): synthetic counterfact generation method used, same as 'method' argument in create_explanation_report
        Returns:
            Display Explanation Report Plot
        '''
        # Data Setup
        data = dict(index=['1', '2', '3', '4', '5', '>5'],
                    percentage=list(report_dict.values()))

        source = ColumnDataSource(data)
        columns = [TableColumn(field="index", title="Number of Feature Values Different"),
                   TableColumn(field="percentage", title="Percentage of Datapoints")]

        # HoverTool Setup
        hover = HoverTool(tooltips=[("Number of Feature Values Different", "@index"),
                                    ("Percentage of Datapoints with such Counterfactuals", "@percentage")])

        # Figure Setup
        TOOLS = [hover]
        p = figure(tools=TOOLS,
                   x_range=['1', '2', '3', '4', '5', '>5'],
                   x_axis_label='Number of Feature Values Different in Counterfactual',
                   y_axis_label='Percentage of Datapoints',
                   plot_height=500, plot_width=1000,
                   title="Global Explainability through Counterfactuals",
                   )
        p.vbar(x='index', top='percentage', source=source, width=0.7)
        p.xgrid.grid_line_color = None
        p.xaxis.major_label_orientation = "horizontal"
        p.y_range.start = 0
        st.bokeh_chart(p, use_container_width=True)

    def plot_robustness_scores(self, robustness_dict_for_plot):
        '''
        Function to plot classwise robustness scores
        Args:
            robustness_dict_for_plot (dictionary): Dictionary containing classwise robustness scores and percentage of predicted instances
        Returns:
            Display Robustness Score Plot
        '''
        # Data Setup
        data = dict(classes=list(robustness_dict_for_plot.keys()),
                    robustness_scores=[i[0] for i in list(robustness_dict_for_plot.values())],
                    class_counts=[i[1] for i in list(robustness_dict_for_plot.values())],
                    mins=[min([i[0] for i in list(robustness_dict_for_plot.values())]) for i in
                          range(len(list(robustness_dict_for_plot.keys())))],
                    avgs=[np.mean([i[0] for i in list(robustness_dict_for_plot.values())]) for i in
                          range(len(list(robustness_dict_for_plot.keys())))])
        source = ColumnDataSource(data=data)

        # HoverTool Setup
        hover = HoverTool(tooltips=[("Robustness Score", "@robustness_scores"),
                                    ("Percentage of Datapoints", "@class_counts")])
        TOOLS = [hover]

        # Plot Setup
        p = figure(tools=TOOLS,
                   x_range=list(robustness_dict_for_plot.keys()),
                   x_axis_label='classes',
                   y_axis_label='',
                   plot_height=400, plot_width=900,
                   title="Overall Model Robustness Scores")
        p.vbar(x='classes', top='class_counts', width=0.80, source=source, alpha=0.4, muted_alpha=0.15,
               legend_label='Percentage of Datapoints')
        p.line(x='classes', y='robustness_scores', source=source, color="blue", muted_color="blue", muted_alpha=0.2,
               line_width=2, legend_label='Robustness Score')
        p.circle(x='classes', y='robustness_scores', source=source, color="blue", line_width=3, muted_color="blue",
                 muted_alpha=0.2, legend_label='Robustness Score')
        p.line(x='classes', y='mins', color="green", source=source, line_width=3, muted_color="blue", muted_alpha=0.2,
               legend_label='Minimum Robustness Score')
        p.line(x='classes', y='avgs', color="red", source=source, line_width=3, muted_color="blue", muted_alpha=0.2,
               legend_label='Average Robustness Score')
        p.xgrid.grid_line_color = None
        p.xaxis.major_label_orientation = "horizontal"
        p.legend.location = 'top_right'
        p.legend.click_policy = 'mute'
        p.y_range.start = 0
        st.bokeh_chart(p, use_container_width=True)

    def input_what_if(self, val_data, default_value):
        '''
        Input Feature Values for What if Explainability (Generalization Needed)
        Needs Generalization (Sugesstions: 1. Provide which are numeric and which are categorical features
                                           2. Depending on the provided features dynamically create number inputs for numeric and selectbox for categorical features)
        Args:
            train_data (pandas dataframe): Training/validation data for obtaining column names
            default_value (pandas dataframe): a single instance (usually from validation) using which deafult values will be loaded
        Returns:
            Creates Layout for What-If Analysis
        '''
        input_features = []
        #feature_names = val_data.columns.to_list()
        feature_names = default_value.columns.to_list()
        cola, colb, colc, cold = st.columns(4)
        cols = [cola, colb, colc, cold]
        for i in range(len(feature_names)):
            #print(val_data[feature_names[i]].dtype)
            # if (isinstance(val_data[str(feature_names[i])].min(), float)) == True:
            #     print('the feature is ', feature_names[i])
            #     input_features.append(cols[i % len(cols)].number_input(str(feature_names[i]),
            #                                                            float(val_data[str(feature_names[i])].min()),
            #                                                            float(val_data[str(feature_names[i])].max()),
            #                                                            step=0.1, value=float(
            #             default_value[str(feature_names[i])])))
            # elif (isinstance(val_data[str(feature_names[i])].min(), int)) == True:

            input_features.append(cols[i % len(cols)].number_input(str(feature_names[i]),
                                                                       int(val_data[str(feature_names[i])].min()),
                                                                       int(val_data[str(feature_names[i])].max()),
                                                                       step=1,
                                                                       value=int(default_value[str(feature_names[i])])))
        # Generating Dataframe
        input_dataframe = pd.DataFrame([input_features], columns=default_value.columns.to_list())
        input_series = pd.Series(input_features, index=default_value.columns.to_list())
        return input_dataframe, input_series
