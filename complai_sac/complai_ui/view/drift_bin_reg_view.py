import streamlit as st
#from streamlit_metrics import metric, metric_row
from view.custom_view_util import metric_row
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import ColumnDataSource, Panel, Tabs, CDSView, GroupFilter, FactorRange
from service.drift_service import DriftService
from view.view_utils import show_prediction_label_distribution, plot_prediction_drift_data, plot_summarized_number_of_features_drifted, plot_summarized_feature_wise_drift, plot_feature_attributes, plot_average_prediction

def app(model):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Dirft Analysis of Live Data')
    driftObj = DriftService(model)
    problem_type =driftObj.problem_type

    # Feature Drift Analysis
    st.header('Feature Attribution Drift Analysis')
    checkbox2 = st.checkbox('Show Feature Drift Analysis',
                            help='Show feature drift between reference dataset and live dataset. It also calculates NDCG Score for feature drift detection where if NDCG is less than 90% then a feature drift has occured.')
    if checkbox2 == True:
        feature_list = driftObj.get_feature_names()
        colx, _ = st.columns(2)
        num_display_features = colx.slider('Choose Number of Top Important Features to Display', min_value=1,
                                           max_value=len(feature_list), value=len(feature_list) // 4)
        plot_button = st.button('Generate Feature Attribution Drift Plot')
        if plot_button == True:
            baseline_feature_attributions = driftObj.get_baseline_feature_attribution()
            #print('baseline_feature_attributions in view ', baseline_feature_attributions)
            val_feature_attributions = driftObj.get_val_data_feature_attribution()
            plot_feature_attributes(baseline_feature_attributions, val_feature_attributions,
                                                 num_display_features)
            expander1 = st.expander('NDCG Score For Feature Drift Alert', expanded=True)
            with expander1:
                overall_ndcg_score = driftObj.get_ndcg_scores()
                overall_ndcg_score_msg = "{:.2f} %".format(overall_ndcg_score)
                metric_row({"Overall Drift Sustainability Score": overall_ndcg_score_msg})
                if int(overall_ndcg_score) < 90:
                    st.error('Feature Drift Alert: Detected Feature Drift')
                elif int(overall_ndcg_score) >= 90:
                    st.subheader('Significant Feature Drift Not Detected')