import streamlit as st
from streamlit_metrics import metric, metric_row
from service.fairness_service import FairnessService
from view.view_utils import plot_fairness_values

def app(model):
    st.title('Model Fairness Monitor')
    fairObj = FairnessService(model)
    protected_features = fairObj.protected_features
    if (None not in protected_features):

        st.header('Protected Attributes Correlation Matrix')
        checkshow = st.checkbox('Show Correlation Matrix',
                                 help='Helps in identifying proxy attributes which have correlation with protected attributes')
        if checkshow == True:
            correlation_matrix = fairObj.get_correlation_matrix_df()
            #print('correlation matrix ', correlation_matrix)
            st.table(correlation_matrix)


        st.header('Legal Fairness')
        checkshow1 = st.checkbox('Show Fainress Overview',
                             help='In this analysis, bias analysis against protect attributes can be analyzed')

        #Get protected attribute
        if checkshow1 == True:
            colaaa, colbbb = st.beta_columns(2)
            fairness_metric = colaaa.selectbox('Choose Fairness Metric', options=['Synthetic Disparate Impact', 'Synthetic Flip Test'], index=0)
            protected_feature = colbbb.selectbox('Choose the Protected feature', options=protected_features, index=0)
            bias_plot_button = st.button('Generate Bias Analysis')
            if bias_plot_button == True:
                protected_feature_attribution, protected_min_score = fairObj.get_feature_fairness(protected_feature, fairness_metric)
                expander1 = st.beta_expander('Bias Score for selected Protected Feature', expanded=True)
                with expander1:
                    metric_row({"Overall Bias Score for " + str(protected_feature): "{:.2f} %".format(protected_min_score)})
                plot_fairness_values(protected_feature_attribution, len(protected_feature_attribution.keys()), fairness_metric)


    else:
        st.write("There are no protected attributes in the given dataset. Fairness is Not Applicable")


