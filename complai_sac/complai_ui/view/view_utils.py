import math
import numpy as np
import streamlit as st
import itertools
import random
import matplotlib
import bokeh
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.layouts import widgetbox, Row
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral5
from bokeh.models import ColumnDataSource, Panel, Tabs, CDSView, GroupFilter, FactorRange

def show_prediction_label_distribution(data_dict, label_types, keys, threshold='default'):
    source = ColumnDataSource(data=data_dict)
    dynamic_lt = [("Percentage of " + i, "@" + i) for i in label_types]
    lt = [("Day", "@periods")]
    lt.extend(dynamic_lt)

    # HoverTool Setup
    hover = HoverTool(tooltips=lt)
    # Figure Setup
    TOOLS = [hover]
    if(len(label_types)<=3):
        colors_list = ["#718dbf", "#e84d60", "#35B778"]
    else:
        colors_list = list(matplotlib.colors.cnames.values())
    colors = random.sample(colors_list, k=len(label_types))
    #colors = ["#718dbf", "#e84d60", "#35B778"]
    p = figure(tools=TOOLS,
               x_range=list(keys),
               plot_height=400, plot_width=900,
               title="Summarized Predicted Label Distribution for Threshold: {}".format(threshold))
    p.vbar_stack(label_types, x='periods', width=0.9, color=colors, source=source, legend_label=label_types)
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = "horizontal"
    p.legend.location = 'center_right'
    p.y_range.start = 0
    st.bokeh_chart(p, use_container_width=True)

def plot_prediction_drift_data(report_dictionary):
    '''
    Plots period-wise prediction drift on live data
    Args:
        report_dist_dict (dictionary): Key as Days and values as tuple of (JS Score, avg fraud pred, number of datapoints)
                                        obtained from function call  get_live_prediction_drift(model, baseline_data)
    Returns:
        Plots prediction drift for the entire week with (JS Score as line plot, avg fraud pred as line plot, number of datapoints as bar plot)
    '''
    # Data Setup
    js_list = []
    avg_pred_list = []
    count_list = []
    norm_count_list = []

    for i in list(report_dictionary.keys()):
        js_list.append(report_dictionary[i][0])
        avg_pred_list.append(report_dictionary[i][1])
        count_list.append(report_dictionary[i][2])

    max_count_list = max(count_list)
    for i in range(len(count_list)):
        norm_count_list.append(count_list[i] / max_count_list)

    data_dict = dict(periods=list(report_dictionary.keys()),
                        js_vals=js_list,
                        avg_preds=avg_pred_list,
                        counts=count_list,
                        norm_counts=norm_count_list)

    source = ColumnDataSource(data=data_dict)

    # HoverTool Setup
    hover = HoverTool(tooltips=[("Day", "@periods"),
                                ("Jensen Shannon Divergence", "@js_vals"),
                                ("Average Prediction Score", "@avg_preds"),
                                ("Number of Datapoints Encountered", "@counts")])
    # Figure Setup
    TOOLS = [hover]
    p = figure(tools=TOOLS,
                x_range=list(report_dictionary.keys()),
                x_axis_label='Period',
                y_axis_label='JS Divergence and Average Prediction Score',
                plot_height=400, plot_width=900,
                title="Prediction Drift Plot")
    p.vbar(x='periods', top='norm_counts', width=0.8, source=source, alpha=0.4, muted_alpha=0.15,
               legend_label='Datapoint Count')
    p.line(x='periods', y='js_vals', source=source, color="blue", muted_color="blue", muted_alpha=0.2, line_width=2,
               legend_label='JS Divergence')
    p.circle(x='periods', y='js_vals', source=source, color="blue", line_width=3, muted_color="blue",
                 muted_alpha=0.2, legend_label='JS Divergence')
    p.line(x='periods', y='avg_preds', source=source, color="#718dbf", line_width=2, muted_color="#718dbf",
               muted_alpha=0.2, legend_label='Average Prediction')
    p.circle(x='periods', y='avg_preds', source=source, line_width=3, muted_color="#718dbf", muted_alpha=0.2,
                 legend_label='Average Prediction')
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = "horizontal"
    p.legend.location = 'top_right'
    p.legend.click_policy = "mute"
    p.y_range.start = 0
    st.bokeh_chart(p, use_container_width=True)

def plot_summarized_number_of_features_drifted(count_dictionary):
    '''
    Plots period-wise number of features drifted on live data
    Args:
        drift_count_dict (dictionary): Keys are periods and values are names of features that are drifted on that day obtained fromget_number_of_features_drifted(model, train_dataset, baseline_feature_attribution, method) function call
    Returns:
        Plots number of features drifted per day as bar chart
    '''
    # Data Setup
    number_of_features = []
    for i in list(count_dictionary.values()):
        number_of_features.append(len(i))
    data_dict = dict(periods=list(count_dictionary.keys()),
                     number=number_of_features,
                     features=list(count_dictionary.values()))
    source = ColumnDataSource(data=data_dict)

    columns = [TableColumn(field="periods", title="Day"),
               TableColumn(field="number", title="Number of Features Drifted"),
               TableColumn(field="features", title="Features Drifted")]
    # HoverTool Setup
    hover = HoverTool(tooltips=[("Day", "@periods"),
                                ("Number of Features Drifted", "@number"),
                                ("Features Drifted", "@features")])
    # Figure Setup
    TOOLS = [hover]

    p = figure(tools=TOOLS,
               x_range=list(count_dictionary.keys()),
               plot_height=400, plot_width=900,
               title="Summarized Number of Features Drifted")
    p.vbar(x='periods', top='number', width=0.8, source=source, alpha=0.7)
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = "horizontal"
    p.y_range.start = 0
    st.bokeh_chart(p, use_container_width=True)


def plot_summarized_feature_wise_drift(drift_value_dictionary, feature_name):
    '''
    Plots period-wise feature drift value for a particular chosen feature
    Args:
        drift_value_dictionary (dictionary): Keys are periods and values are feature attributes of live data for the particular period
                                             Obtained from function call get_summary_feature_attribution_distributions(model, train_dataset, feature_name, method)
        feature_name (string): The feature for which we want to plot attribution for the entire week
    Returns:
        Plots the feature attribution values of live data for a particular selected feature throughout the entire week
    '''
    # Data Setup
    average_feature_attribution = np.mean(list(drift_value_dictionary.values()))
    data_dict = dict(periods=list(drift_value_dictionary.keys()),
                     attribution_values=list(drift_value_dictionary.values()),
                     avg_value=[average_feature_attribution for i in
                                range(len(list(drift_value_dictionary.keys())))])
    source = ColumnDataSource(data=data_dict)

    columns = [TableColumn(field="periods", title="Day"),
               TableColumn(field="attribution_values", title="Values"),
               TableColumn(field="avg_value", title="Average Attribution Value")]
    # HoverTool Setup
    hover = HoverTool(tooltips=[("Day", "@periods"),
                                ("Attribution Value", "@attribution_values"),
                                ("Average Attribution Value", "@avg_value")])
    # Figure Setup
    TOOLS = [hover]

    p = figure(tools=TOOLS,
               x_range=list(drift_value_dictionary.keys()),
               plot_height=400, plot_width=900,
               x_axis_label='Period',
               y_axis_label='Feature Attribution',
               title="Summarized Feature Attribution for Feature: {}".format(feature_name))
    p.vbar(x='periods', top='attribution_values', width=0.8, source=source, alpha=0.7)
    p.line(x='periods', y='avg_value', source=source, color="red", line_width=2)
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = "horizontal"
    p.y_range.start = 0
    st.bokeh_chart(p, use_container_width=True)


def plot_feature_attributes(feature_attribution_dict, live_feature_attribution_dict, display_feature_num=30):
    '''
    Plot Drift in Feature Attributes
    Args:
        feature_attribution_dict (dictionary): Dictionary containing keys as feature names and values as aggreagted feature attribution for baseline
        live_feature_attribution_dict (dictionary): Dictionary containing keys as feature names and values as aggreagted feature attribution for live datasets
        display_feature_num (int): Number of Important features to display
    Returns:
        Display Feature Attributio plot
    '''
    #print('feature_attribution_dict ',feature_attribution_dict)
    #print('live_feature_attribution_dict', live_feature_attribution_dict)
    sorted_feature_attribution_dict = dict(
        sorted(feature_attribution_dict.items(), key=lambda item: item[1], reverse=True))
    sorted_feature_attribution_dict = dict(
        itertools.islice(sorted_feature_attribution_dict.items(), display_feature_num))
    live_feature_attribution_values = []
    for i in list(sorted_feature_attribution_dict.keys()):
        live_feature_attribution_values.append(live_feature_attribution_dict[i])

    # Data Setup
    types = ['Base', 'Live']
    data = dict(feature_names=list(sorted_feature_attribution_dict.keys()),
                base_attribution_values=list(sorted_feature_attribution_dict.values()),
                live_attribution_values=live_feature_attribution_values)
    x = [(features, values) for features in list(sorted_feature_attribution_dict.keys()) for values in types]
    values = []
    value_1 = list(sorted_feature_attribution_dict.values())
    value_2 = live_feature_attribution_values
    for i in range(len(list(sorted_feature_attribution_dict.values()))):
        values.append(value_1[i])
        values.append(value_2[i])

    source = ColumnDataSource(data=dict(x=x, values=values))

    columns = [TableColumn(field="x", title="Feature Name"),
               TableColumn(field="values", title="Values")]
    # HoverTool Setup
    hover = HoverTool(tooltips=[("Feature Name", "@x"),
                                ("Attribution Value", "@values")])
    # Figure Setup
    TOOLS = [hover]
    # Spectral5
    pallete = bokeh.palettes.Spectral5
    p = figure(tools=TOOLS,
               x_range=FactorRange(*x),
               y_axis_label='Feature Attribution',
               plot_height=400, plot_width=display_feature_num * 100,
               title="Feature Attribution")

    p.vbar(x='x', top='values', width=0.92, source=source,
           fill_color=factor_cmap('x', palette=pallete, factors=types, start=1, end=2))
    p.xgrid.grid_line_color = None
    if display_feature_num < 12:
        p.xaxis.major_label_orientation = "horizontal"
    elif display_feature_num >= 12 and display_feature_num < 18:
        p.xaxis.major_label_orientation = math.pi / 4
    elif display_feature_num >= 18:
        p.xaxis.major_label_orientation = "vertical"
        p.xaxis.group_label_orientation = "vertical"
    p.y_range.start = 0
    st.bokeh_chart(p, use_container_width=True)

#Regression specific util

def plot_average_prediction(report_dictionary):
    '''
    Plots period-wise prediction drift on live data
    Args:
        report_dist_dict (dictionary): Key as Days and values as tuple of (JS Score, avg fraud pred, number of datapoints)
                                       obtained from function call  get_live_prediction_drift(model, baseline_data)
    Returns:
        Plots daywise average prediction for the entire week
    '''
    # Data Setup
    avg_pred_list = []

    for i in list(report_dictionary.keys()):
        avg_pred_list.append(round(report_dictionary[i][1] * 1000))

    data_dict = dict(periods=list(report_dictionary.keys()),
                     avg_preds=avg_pred_list)

    source = ColumnDataSource(data=data_dict)

    # HoverTool Setup
    hover = HoverTool(tooltips=[("Day", "@periods"),
                                ("Average Prediction Price", "@avg_preds")])
    # Figure Setup
    TOOLS = [hover]
    p = figure(tools=TOOLS,
               x_range=list(report_dictionary.keys()),
               x_axis_label='Period',
               y_axis_label='Average Prediction Price',
               plot_height=400, plot_width=900,
               title="Prediction Drift Plot")
    p.line(x='periods', y='avg_preds', source=source, color="#718dbf", line_width=2, muted_color="#718dbf",
           muted_alpha=0.2, legend_label='Average Prediction Score')
    p.circle(x='periods', y='avg_preds', source=source, line_width=3, muted_color="#718dbf", muted_alpha=0.2,
             legend_label='Average Prediction Score')
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = "horizontal"
    p.legend.location = 'top_right'
    p.legend.click_policy = "mute"
    p.y_range.start = np.min(avg_pred_list) - 250
    st.bokeh_chart(p, use_container_width=True)


def plot_overall_model_report(report_dictionary, compliance_thresholds):
    '''
    Plots overall model score report
    Args:
        report_dictionary (dictionary): Result Dictionary Containing all the scores for explainability, robustness, fariness, AI Fariness etc. (Total 7 keys should be there)
        compliance_thresholds (dictionary): Same length dictionary having same keys as that of the report_dictionary with compliance preset required scores
    Returns:
        Overall model report plot with compliance scores
    '''
    types = ['Score', 'Threshold']
    x = [(scores, values) for scores in list(report_dictionary.keys()) for values in types]
    values = []
    for i in list(report_dictionary.keys()):
        values.append(report_dictionary[i])
        values.append(compliance_thresholds[i])
    source = ColumnDataSource(data=dict(metric_=x, values=values))

    # HoverTool Setup
    hover = HoverTool(tooltips=[("Metric", "@metric_"),
                                ("Score", "@values")])

    # Figure Setup
    TOOLS = [hover]
    p = figure(tools=TOOLS,
               x_range=FactorRange(*x),
               x_axis_label='Model Metrics',
               y_axis_label='Model Metric Scores',
               plot_height=400, plot_width=900,
               title="Overall Model Score Summary")
    p.vbar(x='metric_', top='values', width=0.92, source=source,
           fill_color=factor_cmap('metric_', palette=Spectral5, factors=types, start=1, end=2))
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = "horizontal"
    p.legend.location = 'top_right'
    p.y_range.start = 0
    st.bokeh_chart(p, use_container_width=True)


def plot_model_comparison(types, model_scores_dict):
    '''
    Plot Model Comparison Scores
    Args:
        model_scores_dict (dictionary): Result Dictionary Containing all the scores for explainability, robustness, fariness, AI Fariness etc. (Total 7 keys should be there)
    Returns:
        Plots model scores comparison graph
    '''
    # Data Setup
    xgb_vals = []
    lr_vals = []
    for i in list(model_scores_dict.keys()):
        xgb_vals.append(model_scores_dict[i][0])
        lr_vals.append(model_scores_dict[i][1])

    #types = ['XGBoost', 'Logistic']
    data = dict(feature_names=list(model_scores_dict.keys()),
                xgboost_values=xgb_vals,
                logistic_Values=lr_vals)
    x = [(scores, values) for scores in list(model_scores_dict.keys()) for values in types]

    values = []
    for i in range(len(list(model_scores_dict.keys()))):
        values.append(xgb_vals[i])
        values.append(lr_vals[i])
    source = ColumnDataSource(data=dict(x=x, values=values))

    # HoverTool Setup
    hover = HoverTool(tooltips=[("Metric", "@x"),
                                ("Score", "@values")])
    # Figure Setup
    TOOLS = [hover]

    p = figure(tools=TOOLS,
               x_range=FactorRange(*x),
               x_axis_label='Model Metrics',
               y_axis_label='Model Metric Scores',
               plot_height=400, plot_width=900,
               title="Overall Model Performance Comparison")
    p.vbar(x='x', top='values', width=0.92, source=source,
           fill_color=factor_cmap('x', palette=Spectral5, factors=types, start=1, end=2))
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = 'horizontal'
    p.y_range.start = 0
    st.bokeh_chart(p, use_container_width=True)


def plot_global_feature_importance(feature_attribution_dict, display_feature_num, method):
    '''
    Shap Global Feature Plot
    Args:
        feature_attribution_dict (dictionary): Dictionary containing keys as feature names and values as aggreagted feature attribution
        display_feature_num (int): Number of features to display in the plot
        method (string): Method used to calculate feature_attribution_dict in function call get_feature_attribution(shap_values, method, feature_names)
    Returns:
        plots global feature importance of a particular model
    '''
    # Data Setup
    sorted_feature_attribution_dict = dict(
        sorted(feature_attribution_dict.items(), key=lambda item: item[1], reverse=True))
    sorted_feature_attribution_dict = dict(
        itertools.islice(sorted_feature_attribution_dict.items(), display_feature_num))

    data = dict(feature_names=list(sorted_feature_attribution_dict.keys()),
                attribution_values=list(sorted_feature_attribution_dict.values()))
    source = ColumnDataSource(data)
    columns = [TableColumn(field="feature_names", title="Feature Name"),
               TableColumn(field="attribution_values", title="Global Importance Value")]

    # HoverTool Setup
    hover = HoverTool(tooltips=[("Feature Name", "@feature_names"),
                                ("Global Importance Value", "@attribution_values")])

    # Figure Setup
    TOOLS = [hover]
    p = figure(tools=TOOLS,
               x_range=list(sorted_feature_attribution_dict.keys()),
               y_axis_label='Feature Attribution',
               x_axis_label='Feature Names',
               plot_height=500, plot_width=display_feature_num * 100,
               title="Global Feature Importance (Method: " + str(method) + " )")
    p.vbar(x='feature_names', top='attribution_values', source=source, width=0.7)
    p.xgrid.grid_line_color = None
    if display_feature_num < len(list(feature_attribution_dict.keys())) // 2:
        p.xaxis.major_label_orientation = "horizontal"
    else:
        p.xaxis.major_label_orientation = "vertical"
    p.y_range.start = 0
    st.bokeh_chart(p, use_container_width=True)

def plot_fairness_values(feature_attribution_dict, display_feature_num, method, settings='auto'):
    '''
    Shap Global Feature Plot
    Args:
        feature_attribution_dict (dictionary): Dictionary containing keys as feature names and values as aggreagted feature attribution
        display_feature_num (int): Number of features to display in the plot
        method (string): Method used to calculate feature_attribution_dict in function call get_feature_attribution(shap_values, method, feature_names)
    Returns:
        plots global feature importance of a particular model
    '''
    # Data Setup
    plot_height = 500
    width = 0.7
    if settings == 'custom':
        plot_height = 250
        width = 0.25
    sorted_feature_attribution_dict = dict(
        sorted(feature_attribution_dict.items(), key=lambda item: item[1], reverse=True))
    sorted_feature_attribution_dict = dict(
        itertools.islice(sorted_feature_attribution_dict.items(), display_feature_num))

    data = dict(sub_group=list(sorted_feature_attribution_dict.keys()),
                bias_score=list(sorted_feature_attribution_dict.values()))
    source = ColumnDataSource(data)
    columns = [TableColumn(field="sub_group", title="Sub Group Name"),
               TableColumn(field="bias_score", title="Bias Score")]

    # HoverTool Setup
    hover = HoverTool(tooltips=[("Sub Group Name", "@sub_group"),
                                ("Bias Score", "@bias_score")])

    # Figure Setup
    TOOLS = [hover]
    p = figure(tools=TOOLS,
               x_range=list(sorted_feature_attribution_dict.keys()),
               y_axis_label='Bias Score',
               x_axis_label='Sub Groups',
               plot_height=plot_height, plot_width=display_feature_num * 100,
               title="Fairness Analysis (Method: " + str(method) + " )")
    p.vbar(x='sub_group', top='bias_score', source=source, width=width)
    p.xgrid.grid_line_color = None
    if display_feature_num < len(list(feature_attribution_dict.keys())) // 2:
        p.xaxis.major_label_orientation = "horizontal"
    else:
        p.xaxis.major_label_orientation = "vertical"
    p.y_range.start = 0
    st.bokeh_chart(p, use_container_width=True)