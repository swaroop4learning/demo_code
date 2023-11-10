import os
import streamlit as st
#from streamlit_metrics import metric, metric_row
from view.custom_view_util import metric_row
from service.summary_service import SummaryService
from view.view_utils import plot_overall_model_report, plot_model_comparison

def app(model):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Overall Model Summary')
    models_list = os.listdir(os.environ['complai_home'])
    #This condition is to remove any OS specific default folders
    if ('.DS_Store' in models_list):
        models_list.remove(('.DS_Store'))
    if ('complai_ui' in models_list):
        models_list.remove(('complai_ui'))
    if ('README.md' in models_list):
        models_list.remove(('README.md'))
    if ('complai_scan-0.1.0.tar.gz' in models_list):
        models_list.remove(('complai_scan-0.1.0.tar.gz'))
    if('complai_scan-0.1.1-mlflow-integration.tar.gz' in models_list):
        models_list.remove(('complai_scan-0.1.1-mlflow-integration.tar.gz'))

    summaryObj = SummaryService(model)

    # Model Loading
    #train_data, train_label_df, val_data, val_label_df = app_features.load_datasets()


    st.header('Global Model Scores')
    checkexp = st.checkbox('Show Global Model Scores for Individual Models',
                           help='Calculate and Show Global Explainability Report, Robustness Score, Drift Score, Model Performance Score and Overall AI Trust Score')
    if checkexp == True:
        generate_report = st.button("Generate Report")
        if generate_report == True:
            #with st.spinner('Generating Explaination and Robustness Report'):
            result_dict = summaryObj.get_trust_scores()
            ai_trust_score_val = summaryObj.ai_trust_score(result_dict)
            final_ai_trust_score_val = "{:.2f}".format(ai_trust_score_val)
            result_dict['AI Trust Factor'] = ai_trust_score_val
            #print('results are ', result_dict)
            exp_expander = st.expander('Overall Model Scores', expanded=True)
            with exp_expander:
                compliance_thresholds = summaryObj.get_trust_scores_thresholds()
                plot_overall_model_report(result_dict, compliance_thresholds)
                metric_row({"AI Trust Factor": final_ai_trust_score_val})
                fix_score = lambda x : "NA" if (x is None) else x
                #print(fairness_score, fairness_score())
                metric_row({"Performance": result_dict['performance_score'], "Explainability": result_dict['explainability_score'],
                             "Robustness": result_dict['robustness_score'],"Drift Sustainability Score": fix_score(result_dict['drift_sustainability_score']),
                            "Fairness": fix_score(result_dict['fairness_score'])})
            #compliance_dict = app_features.load_comppliance_thresholds(model_type)
            compliant, non_compliant = summaryObj.check_compliance(result_dict, compliance_thresholds)
            exp_expander_2 = st.expander('Compliance Summary', expanded=True)
            with exp_expander_2:
                cola, colb = st.columns(2)
                cola.subheader('Compliant Metrices')
                for data in compliant:
                    cola.write('• Metric: ' + str(data[0]) + ', Gain in Score: ' + str(data[1]))
                colb.subheader('Non-Compliant Metrices')
                for data in non_compliant:
                    colb.write('• Metric: ' + str(data[0]) + ', Loss in Score: ' + str(data[1]))

    # Overall Model Comparisons
    st.header('Model Overall Score Comparison')
    checklast = st.checkbox('Show Global Model Scores Comparsion',
                            help='Calculate and Show Global Explainability Report, Robustness Score, Drift Score, Model Performance Score and Overall AI Trust Score for different models in a comparative manner')
    if checklast == True:
        colaaa, colbbb = st.columns(2)
        model2 = colaaa.selectbox("Choose Model for comparision", options=models_list,
                                      index=0)
        generate_comparision_report = st.button("Generate Comparision Report")
        if generate_comparision_report == True:
            if (model == model2):
                st.write('Model 1 and Model2 cannot be same')
            else:
                model_scores_dict = {}
                result_dict = summaryObj.get_trust_scores()
                summaryObj2 = SummaryService(model2)
                result_dict2 = summaryObj2.get_trust_scores()
                ai_trust_score_val1 = summaryObj.ai_trust_score(result_dict)
                result_dict['AI Trust Factor'] = ai_trust_score_val1
                ai_trust_score_val2 = summaryObj2.ai_trust_score(result_dict2)
                result_dict2['AI Trust Factor'] = ai_trust_score_val2
                model_scores_dict = {}
                for i in result_dict:
                    model_scores_dict[i] = []
                    model_scores_dict[i].append(result_dict[i])
                    model_scores_dict[i].append(result_dict2[i])
                #print('model score dict is ', model_scores_dict)
                model_types = [model, model2]
                for i in range(len(model_types)):
                    exp_model = st.expander('Overall Model Scores: ' + str(model_types[i]), expanded=True)
                    with exp_model:
                        final_ai_trust_score_val = "{:.2f}".format(model_scores_dict['AI Trust Factor'][i])
                        final_exp_score = "{:.2f}".format(model_scores_dict['explainability_score'][i])
                        final_robustness = "{:.2f}".format(model_scores_dict['robustness_score'][i])
                        fix_score = lambda x: "NA" if (x is None) else x
                        final_fairness_score = fix_score(model_scores_dict['fairness_score'][i])
                        #final_non_influene_score = "{:.2f}".format(model_scores_dict['NonInfluence Score'][i])
                        #final_drift_score = "{:.2f}".format(model_scores_dict['drift_sustainability_score'][i])
                        final_drift_score = fix_score(model_scores_dict['drift_sustainability_score'][i])
                        final_performance_score = "{:.2f}".format(model_scores_dict['performance_score'][i])
                        metric_row({"AI Trust Factor": final_ai_trust_score_val}),
                        metric_row({"Performance": final_performance_score,
                                    "Explainability": final_exp_score,
                                    "Drift Sustainability Score": final_drift_score,
                                    "Robustness": final_robustness,
                                    "Fairness": final_fairness_score})
                new_expander = st.expander('Overall Model Comparison', expanded=True)
                with new_expander:
                    plot_model_comparison(model_types, model_scores_dict)
