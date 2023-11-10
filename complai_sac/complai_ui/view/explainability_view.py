import streamlit as st
#from streamlit_metrics import metric, metric_row
from view.custom_view_util import metric_row
from service.explainability_service import ExplainabilityService
from view.view_utils import plot_global_feature_importance


def app(model):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Explainations of Prediction')
    explainObj = ExplainabilityService(model)
    problem_type = explainObj.problem_type
    target_classes = explainObj.target_classes
    id_column = explainObj.id_column
    #target_classes_lt = list(target_classes.keys())
    # Model Loading
    val_df, val_label_df, val_pred_df, val_pred_proba_df, val_data = explainObj.load_datasets()
    if id_column is None:
        id_column = 'idx'
    idx_dict = dict(zip(getattr(val_df, id_column), val_df.idx))

    # Prediction Probability Plot
    st.header('Prediction Probability Plot')
    checkbox1 = st.checkbox('Show Prediction Probability Plot',
                            help='Show the interactive prediction probability plot which is constructed directly from database.')
    if checkbox1 == True:
        #data_pred_label, data_probs = explainObj.predict_fraud(loaded_model, data_input)
        if(problem_type=='regression'):
            cola, colb = st.columns(2)
            bins = cola.number_input('Choose Bins', min_value=2, max_value=50, value=10)
            y_preds = val_pred_df.values
            pred_bin_dict, bin_thresholds = explainObj.get_bins_from_pred(y_preds, int(bins))
            render_plot = st.button("Generate Prediction Plot")
            if render_plot == True:
                explainObj.plot_prediction_histogram(pred_bin_dict, bin_thresholds)

        else:
            cola, colb = st.columns(2)
            if(problem_type=='multiclass'):
                target_feature = cola.selectbox('Choose Target Class', options=list(target_classes.keys()), index=1)
                val_pred_proba_target_feature = val_pred_proba_df.apply(lambda x : x[target_classes[target_feature]])
                explainObj.render_plot_widget(val_df, val_pred_df, val_pred_proba_target_feature, target_feature)
            else:
                explainObj.render_plot_widget(val_df, val_pred_df, val_pred_proba_df,
                                              explainObj.get_prefered_class_label())
    # Global Feature Importance
    st.header('Global Feature Importance')
    checkboxglo = st.checkbox('Show Global Feature Importance',
                              help='Calculate and Show Top Chosen Number of Feature Importances of the model on the provided dataset. You can choose the number of features to be displayed.')
    if checkboxglo == True:
        colx, coly = st.columns(2)
        feature_list = explainObj.get_feature_names()
        target_feature = None
        num_display_features = colx.slider('Choose Number of Top Important Features to Display', min_value=1,
                                           max_value=len(feature_list), value=len(feature_list) // 4)
        #global_importance_method = coly.selectbox('Choose Method for Finding Global Feature Importance',
        #                                          options=['Aggregated SHAP', 'Counterfactual Method'], index=0)
        if (problem_type == 'multiclass'):
            target_feature = coly.selectbox('Choose Target Class for feature importance', options=list(target_classes.keys()), index=1)
        global_plot_button = st.button('Generate Global Feature Importance Plot')
        if global_plot_button == True:
            #if global_importance_method == 'Aggregated SHAP':
            #    feature_attributions = explainObj.get_baseline_feature_attribution(target_feature)
                #explainObj.plot_global_feature_importance(feature_attributions, num_display_features)
            #else:
                #feature_attributions = explainObj.feature_importance_CF(num_display_features)
            feature_attributions = explainObj.feature_importance_CF(num_display_features)
            plot_global_feature_importance(feature_attributions, num_display_features, "Counterfactual Method")



    # Global Explainability and Robustness
    st.header('Global Explainability and Robustness')
    checkexp = st.checkbox('Show Global Explainability and Robustness Analysis',
                           help='Calculate and Show Global Explainability Report, Explainability and Robustness Score based on generated nearest counterfactuals')
    if checkexp == True:
        generate_report = st.button("Generate Report")
        if generate_report == True:
            with st.spinner('Generating Explaination and Robustness Report'):
                report, exp_score = explainObj.get_explainability_scores()
                # Unoptimized Slow Code
                # n_robustness = app_features.normalized_robustness_score(loaded_model_np, val_data, robustness, dist_method)
            explainObj.plot_explanation_report(report)
            if(problem_type=='regression'):
                avg_robustness = explainObj.get_regression_robustness_score()
                exp_score = "{:.2f} %".format(exp_score)
                # avg_robustness = avg_robustness * 100
                avg_robustness = "{:.2f} %".format(avg_robustness)
                exp_expander = st.expander('Overall Explainablity and Robustness Score', expanded=True)
                with exp_expander:
                    metric_row({"Explainability Score": exp_score, "Robustness Score": avg_robustness})
            else:
                avg_robustness, min_robustness, robustness_dict_for_plot = explainObj.get_robustness_scores()
                exp_score = "{:.2f} %".format(exp_score)
                #avg_robustness = avg_robustness * 100
                avg_robustness = "{:.2f} %".format(avg_robustness)
                #min_robustness = min_robustness * 100
                min_robustness = "{:.2f} %".format(min_robustness)
                exp_expander = st.expander('Overall Model Explainability Score', expanded=True)
                with exp_expander:
                    metric_row({"Explainability Score": exp_score})
                rbst_expander = st.expander('Overall Model Robustness Score', expanded=True)
                with rbst_expander:
                    explainObj.plot_robustness_scores(robustness_dict_for_plot)
                    metric_row({"Average Robustness Score": avg_robustness, "Minimum Robustness Score": min_robustness})

    # Counterfactual Witget
    st.header('Nearest Counterfactual Finder')
    checkfind = st.checkbox('Show Nearest Counterfactual Finder from Database',
                            help='Nearest Counterfactual means a datapoint which is nearest in terms of vector distance from the chosen datapoint but has an opposite predicted label. It can be of two types, real nearest counterfactual where the counterfactual is chosen from the provided dataset and synthetic counterfactual which is the closest possible counterfactual generated using genetic algorithm.')
    if checkfind == True:
        colaa, colbb = st.columns(2)
        selectindex = colaa.selectbox("Choose Datapoint Index", options=list(idx_dict.keys()),
                                      key='Counter_Select')
        counterfact_button = st.button('Calculate Nearest (Real and Synthetic) Counterfactuals')
        if counterfact_button == True:
            # data_pred_label, data_probs = app_features.predict_fraud(loaded_model, data_input)
            #data_pred_label, data_probs = app_features.custom_predict(loaded_model, data_input, cf_threshold)
            #counterfact_idx, counterfact = app_features.nearest_counterfactual(selectindex, data_input, data_pred_label,
                                                                               #distance_mode)
            instance, pred_label, pred_prob = explainObj.get_val_data_by_index(val_df, idx_dict[selectindex])
            real_counterfact = explainObj.get_real_counterfactuals(idx_dict[selectindex])
            synthetic_counterfact = explainObj.get_synthetic_counterfactuals(idx_dict[selectindex])
            st.subheader('Chosen Data Point')
            expander1 = st.expander('Data Point Details', expanded=True)


            with expander1:
                st.dataframe(instance)
                st.write('Datapoint ID: ', selectindex)
                if(problem_type=="regression"):
                    pred_label = "{:.2f}".format(pred_label)
                    st.write('Predicted Label: ', pred_label)
                else:
                    st.write('Predicted Label: ', pred_label)
                    st.write(str(pred_label)+' Probability:', round(float(pred_prob), 3))
            st.subheader('Nearest Real Counterfactual')
            expander2 = st.expander('Nearest Real Counterfactual Details', expanded=True)
            with expander2:
                st.dataframe(real_counterfact)
                st.write('Counterfactual label: NOT ', pred_label)
                # st.write('Datapoint ID: ', counterfact_idx)
                # if data_pred_label[counterfact_idx] == 0:
                #     st.write('Predicted Label: Not Fraud')
                # elif data_pred_label[counterfact_idx] == 1:
                #     st.write('Predicted Label: Potential Fraud')
                # st.write('Fraud Probability:', round(float(data_probs[counterfact_idx]), 3))
            st.subheader('Nearest Possible Counterfactual (Synthetic)')
            expander3 = st.expander('Nearest Synthetic Counterfactual Details', expanded=True)
            with expander3:
                st.dataframe(synthetic_counterfact)
                st.write('Counterfactual label: NOT ',pred_label)
                st.subheader('Changed Feature Values from Chosen Datapoint to Synthetic Counterfactual')
                st.write(
                    'To change the original result to the target result, all the following features have to be changed to meet the counterfactual result')
                explainObj.return_different_features(instance, synthetic_counterfact, display=True)

    # What If Analysis
    st.header('What If Analysis')
    checkwit = st.checkbox('Launch What If Analysis',
                           help='In this analysis, a real datapoint can be selected it can be edited. After editing, the corresponding explanation can be visualized for the edited datapoint.')
    if checkwit == True:
        cola, colb = st.columns(2)
        index = cola.selectbox("Choose Datapoint Index", options=list(idx_dict.keys()))
        # instance = app_features.query(database, int(index))
        instance, pred_label, pred_prob = explainObj.get_val_data_by_index(val_df, idx_dict[index])
        if instance is not None:
            st.write('Chosen Data Point')
            st.dataframe(instance)
            checkedit = st.checkbox('Edit Datapoint for What If Analysis')
            if checkedit == True:
                try:
                    input_dataframe, input_series = explainObj.input_what_if(val_df, instance)
                    prediction_button = st.button('Predict')
                    if prediction_button == True:
                        st.error('Please define your model endpoint in config for enabling What-If feature')
                except:
                     st.error('Please define your model endpoint in config for enabling What-If feature')
                    # st.subheader('Prediction Result')
                    # expanderwhat = st.expander('What-If Explanation', expanded=True)
                    # with expanderwhat:
                    #     pred_class, pred_prob = app_features.predict_instance(loaded_model, input_dataframe,
                    #                                                           cf_threshold_wit)
                    #     # pred_class = loaded_model.predict(input_dataframe)
                    #     # pred_prob = loaded_model.predict_proba(input_dataframe)
                    #     predicted_class = None
                    #     predicted_prob = None
                    #     if pred_class == 0:
                    #         predicted_class = 'Not Fraud'
                    #     elif pred_class == 1:
                    #         predicted_class = 'Potential Fraud'
                    #     predicted_prob = pred_prob
                    #     st.write("Prediction: ", predicted_class)
                    #     st.write("Probability: ", round(float(predicted_prob), 3))
                    #     explainer = app_features.explainer_create(loaded_model, train_data)
                    #     local_shap_values = app_features.get_shap_values(explainer, input_dataframe)
                    #     st.subheader('Waterfall Plot')
                    #     st.pyplot(app_features.waterfall_plot(local_shap_values))
                    # new_synthetic_counterfact = app_features.nearest_synthetic_counterfactual(loaded_model_np,
                    #                                                                           train_data,
                    #                                                                           train_label_df,
                    #                                                                           input_dataframe,
                    #                                                                           cf_method,
                    #                                                                           cf_threshold_wit)
                    # new_synthetic_label, new_synthetic_prob = app_features.predict_instance(loaded_model,
                    #                                                                         new_synthetic_counterfact,
                    #                                                                         cf_threshold_wit)
                    # expander4 = st.expander('Nearest Synthetic Counterfactual to Custom Datapoint', expanded=True)
                    # with expander4:
                    #     st.dataframe(new_synthetic_counterfact)
                    #     if new_synthetic_label == 0:
                    #         st.write('Predicted Label: Not Fraud')
                    #     elif new_synthetic_label == 1:
                    #         st.write('Predicted Label: Potential Fraud')
                    #     st.write('Fraud Probability:', round(float(new_synthetic_prob[0]), 3))
                    #     st.subheader('Changed Feature Values from Chosen Datapoint to Synthetic Counterfactual')
                    #     st.write(
                    #         'To change the original result to the target result, all the following features have to be changed to meet the counterfactual result')
                    #     app_features.return_different_features(input_dataframe, new_synthetic_counterfact, display=True)