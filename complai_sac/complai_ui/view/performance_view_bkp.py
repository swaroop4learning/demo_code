import streamlit as st
from streamlit_metrics import metric, metric_row
from mlxtend.plotting import plot_confusion_matrix
from service.model_performance_service import PerformanceFeatures


def app(model):
    st.title('Model Performance Monitor')

    perfObj = PerformanceFeatures(model)
    problem_type = perfObj.problem_type
    # Model Loading
    val_df, val_label_df, val_pred_df, val_pred_proba_df, val_data = perfObj.load_datasets()

    # Confusion Matrix and Model Performance Statistics
    st.header('Model Performance Statistics and Confusion Matrix')
    checkshow1 = st.checkbox('Show Model Performance Metric',
                             help='In this analysis, the decision probability threshold can be set and model performance can be analyzed based on chosen threshold.')
    if checkshow1 == True:
        colb, _ = st.beta_columns(2)
        #loaded_model, loaded_model_np = app_features.load_pretrained_model(model_type)
        threshold_value=None
        if(problem_type == 'binary-classification'):
            threshold_value = colb.slider('Set Threshold Probability', min_value=0.0, max_value=1.0, step=0.05, value=0.5)
        thres_button = st.button('Generate Performance Report',
                                 help='Choose threshold probability, greater than and equal which a datapoint will be classified as Positive Class, less than which the datapoint will be classified as Negative Class')
        if thres_button == True:
            y_val = val_label_df
            y_pred_label = val_pred_df
            if(problem_type == 'regression'):
                expander1 = st.beta_expander('Performance Metric Results', expanded=True)
                with expander1:
                    metrics = perfObj.regression_performance_report(y_val, y_pred_label)
                    metric_row(metrics)
            else:
                if(problem_type == 'binary-classification'):
                    y_pred_label = perfObj.custom_predict(val_pred_proba_df, threshold = threshold_value)
                expander1 = st.beta_expander('Performance Metric Results', expanded=True)
                with expander1:
                    metrics = perfObj.model_performance_report(y_val, y_pred_label)
                    metric_row(metrics)
                expander2 = st.beta_expander('Confusion Matrix', expanded=True)
                with expander2:
                #perfObj.show_confusion_matrix(y_val, y_pred_label)
                    cm, target_names = perfObj.show_confusion_matrix(y_val, y_pred_label)
                    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                                show_absolute=True,  # Absolute Counts for FPR, TPR, TNR, FNR
                                                show_normed=True,  # Percentages for FPR, TPR, TNR, FNR
                                                colorbar=False,  # Show Colorbar for count - color intensity mapping
                                                class_names=target_names)
                    st.pyplot(fig, use_columnwidth=False)


    # Model Performance on Slices
    st.header('Model Performance on Slices')
    checkslice = st.checkbox('Launch Model Performance on Slices',
                             help='Check Model performance on different subsets of dataset by creating slices and setting custom threshold.')
    if checkslice == True:
        # model_type = st.selectbox('Choose Pretrained Model for Slice Based Analysis',
        #                           options=['XGBoost', 'Logistic Regression'])
        # loaded_model, loaded_model_np = perfObj.load_pretrained_model(model_type)
        colaaa, colbbb, colccc, colddd, coleee = st.beta_columns(5)
        #feature_names = perfObj.get_feature_names()[:-1]
        feature_names = perfObj.get_feature_names()
        chosen_feature_1 = colaaa.selectbox('Select Primary Feature', options=feature_names,
                                            index=len(feature_names) - 1)
        chosen_bucket_1 = colbbb.selectbox('Select Primary Feature Bucket', options=[i for i in range(1, 11)])
        chosen_feature_2 = colccc.selectbox('Select Secondary Feature', options=feature_names,
                                            index=len(feature_names) - 1)
        chosen_bucket_2 = colddd.selectbox('Select Secondary Feature Bucket', options=[i for i in range(1, 11)])
        threshold=None
        if(problem_type=='binary-classification'):
            threshold = coleee.slider('Select Threshold Probability', min_value=0.0, max_value=1.0, step=0.05, value=0.5)
        predict_slices = st.button('Generate Slices and Predict Performance on Slices')
        if predict_slices == True:
            with st.spinner('Generating Slices and Running Model..'):
                slice_list, slice_label_list, pred_proba_list, report_dict = perfObj.create_slices(val_df, chosen_feature_1,
                                                                                       chosen_feature_2,
                                                                                       int(chosen_bucket_1),
                                                                                       int(chosen_bucket_2))
                if (slice_list != None and slice_label_list != None and report_dict != None):
                    slice_expander = st.beta_expander('Model Performance Report on Generated Slices', expanded=True)
                    with slice_expander:
                        if(problem_type=='regression'):
                            report_dataframe = perfObj.predict_slices_regression(slice_list, slice_label_list, pred_proba_list,
                                                                      report_dict, threshold)
                        else:
                            report_dataframe = perfObj.predict_slices(slice_list, slice_label_list, pred_proba_list,
                                                                       report_dict, threshold)
                        st.table(report_dataframe.assign(index='').set_index('index'))
