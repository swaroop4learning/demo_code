import streamlit as st
from streamlit_metrics import metric, metric_row
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import ColumnDataSource, Panel, Tabs, CDSView, GroupFilter, FactorRange
from service.drift_multi_class_service import DriftMultiService
from view.view_utils import show_prediction_label_distribution, plot_prediction_drift_data, plot_summarized_number_of_features_drifted, plot_summarized_feature_wise_drift, plot_feature_attributes, plot_average_prediction

def app(model):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Dirft Analysis of Live Data')
    driftObj = DriftMultiService(model)
    target_classes = list(driftObj.target_classes.keys())
    # Live Label Distribution
    st.header('Live Prediction Label Distribution')
    checkbox_ = st.checkbox('Show Day-wise Live Label Distribution',
                            help='Shows Day-wise label distribution of live data and allows user to set custom probability threshold to analyze how new threshold affects the live distribution of predicted labels.')
    if checkbox_ == True:
        cola, colb = st.beta_columns(2)
        #loaded_model, loaded_model_np = app_features.load_pretrained_model(model_type)
        threshold=None
        live_dist_button = st.button('Generate Daywise Live Label Distribution Summary')
        if live_dist_button == True:
            data_dict = driftObj.get_live_label_distribution(threshold)
            #data_dict, label_types = driftObj.get_summarized_live_label_distribution(label_dist_dict, threshold)
            label_types = list(driftObj.target_classes.keys())
            label_dist_dict_keys = driftObj.periods.values()
            show_prediction_label_distribution(data_dict, label_types, label_dist_dict_keys, threshold)


    # Prediction Drift Analysis
    st.header('Live Prediction Drift Analysis')
    checkboxnew = st.checkbox('Show Day-wise Live Prediction Drift Analysis',
                              help='Shows Day-wise prediction drift of live data.')
    if checkboxnew == True:
        cola, colb = st.beta_columns(2)
        target_class = cola.selectbox('Choose Target Class for Prediction Drift', options=target_classes,
                                      index=1)
        #loaded_model, loaded_model_np = app_features.load_pretrained_model(model_type)
        pred_dist_button = st.button('Generate Daywise Prediction Drift Summary')
        if pred_dist_button == True:
            pred_report = driftObj.get_live_prediction_drift(target_class)
            plot_prediction_drift_data(pred_report)

        # Number of Features Drifting Daily
    st.header('Daywise Number of Features Drifting')
    checkbox0 = st.checkbox('Show Day-wise Number of Features Drifting',
                            help='Shows Day-wise number of features that are drifting based on feature attribution with respect to baseline feature importance.')
    if checkbox0 == True:
        colaa, colbb = st.beta_columns(2)
        target_class = colaa.selectbox('Choose Target Class for Analysis', options=target_classes, index=1)
        feature_plot_button = st.button('Generate Daywise Number of Features Drift Summary')
        if feature_plot_button == True:
            number_dict = driftObj.get_number_of_features_drifted(target_class)
            plot_summarized_number_of_features_drifted(number_dict)

    # Feature-wise Feature Drift Analysis
    st.header('Featurewise Attribution Drift Analysis')
    checkbox1 = st.checkbox('Show Feature-wise Drift Analysis Report',
                            help='Shows Feature-wise attribution drift between reference dataset and live dataset for a particular feature over time.')
    if checkbox1 == True:
        feature_list = driftObj.get_feature_names()
        colw, colx = st.beta_columns(2)
        feature_name = colw.selectbox("Feature to show Drift Analysis", options=feature_list, index=0)
        target_class = colx.selectbox('Target Class for Interest', options=target_classes, index=1)
        feature_plot_button = st.button('Generate Featurewise Attribution Drift Plot')
        if feature_plot_button == True:
            drift_dict = driftObj.get_summary_feature_attribution_distributions(feature_name, target_class)
            plot_summarized_feature_wise_drift(drift_dict, feature_name)

    # Feature Drift Analysis
    st.header('Daywise Feature Attribution Drift Analysis')
    checkbox2 = st.checkbox('Show Daywise Feature Drift Analysis',
                            help='Show Day-wise feature drift between reference dataset and live dataset. It also calculates NDCG Score for feature drift detection where if NDCG is less than 90% then a feature drift has occured.')
    if checkbox2 == True:

        # database = app_features.connect_database()
        # validation_data = app_features.database_to_model_input(database)
        feature_list = driftObj.get_feature_names()
        colx, coly = st.beta_columns(2)
        num_display_features = colx.slider('Choose Number of Top Important Features to Display', min_value=1,
                                           max_value=len(feature_list), value=len(feature_list) // 4)
        target_class_ = coly.selectbox('Target Class for Drift Analysis', options=target_classes, index=1)
        plot_button = st.button('Generate Feature Attribution Drift Plot')
        if plot_button == True:
            baseline_feature_attributions = driftObj.get_baseline_feature_attribution(target_class_)
            #print('baseline_feature_attribution is ', baseline_feature_attributions)
            val_feature_attributions = driftObj.get_val_data_feature_attribution(target_class_)
            plot_feature_attributes(baseline_feature_attributions, val_feature_attributions,
                                    num_display_features)
            expander1 = st.beta_expander('NDCG Score For Feature Drift Alert', expanded=True)
            with expander1:
                #ndcg = driftObj.NDCG_Score(baseline_feature_attributions, val_feature_attributions)
                ndcg_class = driftObj.get_ndcg_class(target_class_)
                print('ndcg_class score is ',ndcg_class)
                overall_ndcg = driftObj.get_ndcg_scores()
                ndcg_score = "{:.2f} %".format(ndcg_class)
                overall_ndcg_score = "{:.2f} %".format(overall_ndcg)
                metric_row({"Drift Score for " + str(target_class_): ndcg_score, "Overall Drift Sustainability Score": overall_ndcg_score})
                if overall_ndcg < 90:
                    st.error('Feature Drift Alert: Detected Feature Drift')
                elif overall_ndcg >= 90:
                    st.subheader('Significant Feature Drift Not Detected')

