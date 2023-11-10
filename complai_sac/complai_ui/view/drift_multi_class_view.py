import streamlit as st
#from streamlit_metrics import metric, metric_row
from view.custom_view_util import metric_row
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import ColumnDataSource, Panel, Tabs, CDSView, GroupFilter, FactorRange
from service.drift_multi_class_service import DriftMultiService
from view.view_utils import show_prediction_label_distribution, plot_prediction_drift_data, \
    plot_summarized_number_of_features_drifted, plot_summarized_feature_wise_drift, plot_feature_attributes, \
    plot_average_prediction


def app(model):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Dirft Analysis of Live Data')
    driftObj = DriftMultiService(model)
    target_classes = list(driftObj.target_classes.keys())
    # Live Label Distribution

    # Feature Drift Analysis
    st.header('Feature Attribution Drift Analysis')
    checkbox2 = st.checkbox('Show Feature Drift Analysis',
                            help='Show feature drift between reference dataset and live dataset. It also calculates NDCG Score for feature drift detection where if NDCG is less than 90% then a feature drift has occured.')
    if checkbox2 == True:

        # database = app_features.connect_database()
        # validation_data = app_features.database_to_model_input(database)
        feature_list = driftObj.get_feature_names()
        colx, coly = st.columns(2)
        num_display_features = colx.slider('Choose Number of Top Important Features to Display', min_value=1,
                                           max_value=len(feature_list), value=len(feature_list) // 4)
        target_class_ = coly.selectbox('Target Class for Drift Analysis', options=target_classes, index=1)
        plot_button = st.button('Generate Feature Attribution Drift Plot')
        if plot_button == True:
            baseline_feature_attributions = driftObj.get_baseline_feature_attribution(target_class_)
            # print('baseline_feature_attribution is ', baseline_feature_attributions)
            val_feature_attributions = driftObj.get_val_data_feature_attribution(target_class_)
            plot_feature_attributes(baseline_feature_attributions, val_feature_attributions,
                                    num_display_features)
            expander1 = st.expander('NDCG Score For Feature Drift Alert', expanded=True)
            with expander1:
                # ndcg = driftObj.NDCG_Score(baseline_feature_attributions, val_feature_attributions)
                ndcg_class = driftObj.get_ndcg_class(target_class_)
                #print('ndcg_class score is ', ndcg_class)
                overall_ndcg = driftObj.get_ndcg_scores()
                ndcg_score = "{:.2f} %".format(ndcg_class)
                overall_ndcg_score = "{:.2f} %".format(overall_ndcg)
                metric_row({"Drift Score for " + str(target_class_): ndcg_score,
                            "Overall Drift Sustainability Score": overall_ndcg_score})
                if overall_ndcg < 90:
                    st.error('Feature Drift Alert: Detected Feature Drift')
                elif overall_ndcg >= 90:
                    st.subheader('Significant Feature Drift Not Detected')

