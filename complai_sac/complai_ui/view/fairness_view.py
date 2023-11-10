import streamlit as st
#from streamlit_metrics import metric, metric_row
from view.custom_view_util import metric_row
from service.fairness_service import FairnessService
from view.view_utils import plot_fairness_values

def app(model):
    st.title('Model Fairness Monitor')
    fairObj = FairnessService(model)
    protected_features = fairObj.protected_features
    supported_metrics = ['Precision', 'f1_score', 'Recall', 'False Positive Rate', 'False Discovery Rate', 'True Negative Rate', 'False Negative Rate']
    if (None not in protected_features):
        val_df, val_label_df, val_pred_df, val_pred_proba_df, val_data = fairObj.load_datasets()
        combinations = fairObj.feature_combinations
        combined_features = protected_features.copy()
        protected_features_mappings = fairObj.protected_features_mappings
        data_df = fairObj.get_decrypted_data(val_df, protected_features_mappings)
        if combinations is not None:
            combined_features, data_df = fairObj.get_protected_data(protected_features.copy(), data_df, combinations)

        st.header('Protected Attributes Correlation Matrix')
        checkshow = st.checkbox('Show Correlation Matrix',
                                 help='Helps in identifying proxy attributes which have correlation with protected attributes')
        if checkshow == True:
            #print('the protected features are before corr ', protected_features)
            try:
                correlation_matrix = fairObj.get_correlation_matrix_df(protected_features)
                #print('correlation matrix ', correlation_matrix)
                st.table(correlation_matrix)
            except:
                st.write('Correlation table not available for this dataset. Please update config and re run scan again')


        st.header('Counterfactual Legal Fairness')
        checkshow1 = st.checkbox('Show Fainress Overview',
                             help='In this analysis, bias analysis against protect attributes can be analyzed with counterfactuals')

        #Get protected attribute
        if checkshow1 == True:
            colaaa, colbbb = st.columns(2)
            fairness_metric = colaaa.selectbox('Choose Fairness Metric', options=['Synthetic Disparate Impact', 'Synthetic Flip Test'], index=0)
            protected_feature = colbbb.selectbox('Choose the Protected feature', options=protected_features, index=0)
            bias_plot_button = st.button('Generate Bias Analysis')
            if bias_plot_button == True:
                protected_feature_attribution, protected_min_score = fairObj.get_feature_fairness(protected_feature, fairness_metric)
                #print('the protected_attribution is ',protected_feature_attribution)
                expander1 = st.expander('Bias Score for selected Protected Feature', expanded=True)
                with expander1:
                    metric_row({"Overall Bias Score for " + str(protected_feature): "{:.2f} %".format(protected_min_score)})
                plot_fairness_values(protected_feature_attribution, len(protected_feature_attribution.keys()), fairness_metric)

        st.header('Legal Fairness')
        checkshow2 = st.checkbox('Show Fairness Overview',
                                 help='In this analysis, bias analysis against protect attributes can be analyzed')
        if checkshow2 == True:
            colaaa1, colbbb1, colccc1 = st.columns(3)
            fairness_metric = colaaa1.selectbox('Choose Fairness Metric',
                                               options=['Disparate Impact', 'Equal Opportunity'], index=0)
            combined_feature = colbbb1.selectbox('Choose Protected feature', options=combined_features, index=0)
            threshold_value = colccc1.slider('Set Threshold Probability', min_value=0.0, max_value=1.0, step=0.05,
                                          value=0.5)
            bias_plot_button1 = st.button('Generate Legal Fairness Analysis')
            if bias_plot_button1 == True:
                legal_score_dict, min_max_score = fairObj.get_legal_scores(data_df, combined_feature, threshold_value, fairness_metric)
                expander2 = st.expander('Legal Score for selected Protected Feature', expanded=True)
                with expander2:
                    metric_row({"Overall Legal Bias Score for " + str(combined_feature): "{:.2f} %".format(min_max_score)})
                plot_fairness_values(legal_score_dict, len(legal_score_dict.keys()),
                                 fairness_metric)

        st.header('Model Fairness')
        checkshow3 = st.checkbox('Show Model Fairness Overview',
                                 help='In this analysis, model bias analysis against protect attributes can be analyzed')
        if checkshow3 == True:
            colaaa2, colbbb2 = st.columns(2)
            combined_feature = colaaa2.selectbox('Choose the Protected features', options=combined_features, index=0)
            threshold_value = colbbb2.slider('Set a Threshold Probability', min_value=0.0, max_value=1.0, step=0.05,
                                             value=0.5)
            bias_plot_button2 = st.button('Generate Model Fairness Analysis')
            if bias_plot_button2 == True:
                model_fairness_metric_dict = fairObj.get_model_fairness(data_df,combined_feature,threshold_value)
                for i in supported_metrics:
                    fairness_dict = {}
                    for sub_group in model_fairness_metric_dict:
                        fairness_dict[sub_group] = model_fairness_metric_dict[sub_group][i]
                    plot_fairness_values(fairness_dict, len(fairness_dict.keys()),
                                         i, 'custom')
    else:
        st.write("There are no protected attributes in the given dataset. Fairness is Not Applicable")
