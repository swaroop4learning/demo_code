import os
import yaml
import pandas as pd
import numpy as np
from data.model_dao import get_db_con, get_performance_data, get_validation_data, get_ndcg_data, get_fairness_data

class SummaryService():

    def __init__(self, model):
        self.model = model
        self.model_home = os.environ['complai_home'] + "/" + model
        policy_yml_path = self.model_home + "/" + model + "_policy.yml"
        with open(policy_yml_path, 'r') as yaml_file:
            self.policy = yaml.safe_load(yaml_file)
        yml_path = self.model_home + "/" + model + "_config.yml"
        with open(yml_path, 'r') as yaml_file:
            self.config = yaml.safe_load(yaml_file)
        db_path = self.model_home+"/data/db.json"
        self.db = get_db_con(db_path)
        self.trust_scores_thresholds = self.policy['trust_scores']
        self.problem_type = self.config['problem_type']
        self.protected_features = self.config['protected_attributes']
        self.fairness_metric = self.config['fairness_metric']
        if (None not in self.protected_features):
            self.protected_features_mappings = self.config['protected_attributes_mappings']
        # print('config is ', self.config)

    def get_explainability_scores(self):
        val_data = get_validation_data(self.db)
        val_df = pd.DataFrame(val_data).dropna()
        explainability_score = val_df['explainability_score'][0]*100
        explainability_score = round(explainability_score, 2)
        return explainability_score

    def get_robustness_score(self):
        val_data = get_validation_data(self.db)
        val_df = pd.DataFrame(val_data).dropna()
        if(self.problem_type == 'regression'):
            min_robustness = val_df['avg_robustness'][0]*100
        else:
            min_robustness = val_df['min_robustness'][0]*100
        min_robustness = round(min_robustness, 2)
        #avg_robustness = val_df['avg_robustness'][0] * 100
        return min_robustness

    def get_drift_score(self):
        drift_data = get_ndcg_data(self.db)
        if(len(drift_data)>1):
            drift_score = min([i['ndcg_score'] for i in drift_data])
        else:
            drift_score = drift_data[0]['ndcg_score']
        drift_score = round(drift_score, 2) * 100
        if(self.problem_type == 'multiclass'):
            drift_score = None
        return drift_score

    def get_feature_fairness(self, protected_feature, fairness_metric):
        fairness_data = get_fairness_data(self.db)[0]
        if(fairness_metric=='Synthetic Flip Test'):
            fairness_feature_data = fairness_data['fairness_info'][protected_feature]
        else:
            fairness_feature_data = fairness_data['di_info'][protected_feature]

        feature_mapping = self.protected_features_mappings[protected_feature]
        #protected_feature_attribution = dict((feature_mapping[key], value) for (key, value) in fairness_feature_data.items())
        protected_feature_attribution = {}
        feature_attributions = []
        for i,v in feature_mapping.items():
            protected_feature_attribution[i] = fairness_feature_data[str(v)]*100
            feature_attributions.append(fairness_feature_data[str(v)]*100)
        if (fairness_metric == 'Synthetic Flip Test'):
            overall_min_protected = min(feature_attributions)
        else:
            min_val = min(feature_attributions)
            max_val = max(feature_attributions)
            overall_min_protected = round((min_val / max_val) * 100, 2)
        return protected_feature_attribution, overall_min_protected

    def get_fairness_score(self):
        fairness_data = get_fairness_data(self.db)
        if(len(fairness_data)==0):
            return None
        else:
            overall_fairness_score = []
            for i in self.protected_features:
                feature_attribution, attribute_fairness_score = self.get_feature_fairness(i,self.fairness_metric)
                overall_fairness_score.append(attribute_fairness_score)
            fairness_score = min(overall_fairness_score) #logic to be defined
            fairness_score = round(fairness_score, 2)
            return fairness_score

    def get_trust_scores_thresholds(self):
        ai_trust_score_val = 0.0
        n = 0
        trust_dict = self.trust_scores_thresholds
        for i in list(trust_dict.keys()):
            if (trust_dict[i] is not None):
                ai_trust_score_val = ai_trust_score_val+trust_dict[i]
            else:
                n = 1
        count = len(list(trust_dict.keys()))-n
        ai_trust_score_val = ai_trust_score_val / count
        trust_dict['AI Trust Factor'] = ai_trust_score_val
        return trust_dict

    def get_trust_scores(self):
        scores_dict = {}
        perf_data = get_performance_data(self.db)
        scores_dict['performance_score'] = round(perf_data[0]['performance_score']*100,2)
        scores_dict['explainability_score'] = self.get_explainability_scores()
        scores_dict['robustness_score'] = self.get_robustness_score()
        try:
            scores_dict['drift_sustainability_score'] = self.get_drift_score()
        except Exception as e:
            scores_dict['drift_sustainability_score'] = None
        scores_dict['fairness_score'] = self.get_fairness_score()
        return scores_dict

    def ai_trust_score(self, result_dict):
        '''
        Generate Overall AI Trust Score for Model. Suggested to use weighted average rather than simple average
        Args:
            result_dict (dictionary): Result Dictionary Containing all the scores for explainability, robustness, fariness etc.
        '''
        ai_trust_score_val = 0
        n = 0
        for i in list(result_dict.keys()):
            if(result_dict[i] is not None):
                ai_trust_score_val += result_dict[i]
            else:
                n += 1
        count =  len(list(result_dict.keys()))-n
        ai_trust_score_val = ai_trust_score_val / count
        return ai_trust_score_val

    def check_compliance(self, results, compliance):
        '''
        Check which model metrices are compliant
        Args:
            results (dictionary): Result Dictionary Containing all the scores for explainability, robustness, fariness, AI Fariness etc. (Total 7 keys should be there)
            compliance (dictionary): Same length dictionary having same keys as that of the report_dictionary with compliance preset required scores
        Returns:
            compliant (list): List of Compliant metrices
            non_compliant (list): List of Non-Compliant metrices
        '''
        compliant = []
        non_compliant = []
        for i in list(results.keys()):
            if(results[i] == None):
                results[i] = 0
            if(compliance[i] == None):
                compliance[i] = 0
            if (results[i] >= compliance[i]):
                abs_val = results[i] - compliance[i]
                abs_val = "{:.2f} %".format(abs_val)
                compliant.append((i, abs_val))
            else:
                abs_val = results[i] - compliance[i]
                abs_val = "{:.2f} %".format(abs_val)
                non_compliant.append((i, abs_val))
        return compliant, non_compliant





