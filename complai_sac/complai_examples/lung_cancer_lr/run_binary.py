import logging
from extendables.get_predictions import GetPredictions
from generate_values.generate_counterfactuals_binary import GenerateCounterfactualsBinary
# from generate_values.generate_shap_values_binary import GenerateShapBinary
from generate_values.generate_performance_values_binary import GenerateBinaryPerformanceScore
from generate_values.generate_fairness_binary import GenerateFairnessBinary
from generate_values.generate_predictions_live import GenerateLivePredictions
from generate_values.generate_drift_score_counterfactual_binary import GenerateCounterfactualFeatureAttributionBinary
from utils.get_data_generic import GenericGetData
from utils.get_connections import tiny_db_connection


class AdultPredictor(GetPredictions):
    def __init__(self):
        super().__init__()

    def get_proba_scores(self, X_data, model):
        probability_scores = model.predict_proba(X_data)
        return probability_scores[:, 1]


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)
    yaml_path = "./lung_cancer_lr_config.yml"
    data_obj = GenericGetData(yaml_path=yaml_path)
    x_train_data, validation_data_new_moified, validation_data_new, live_data_new_moified, y_val, model, config\
        , x_train_data_org =  data_obj.get_data()
    # print(pd.DataFrame(x_train_data))
    validation_data_new_moified_original = validation_data_new_moified.copy()
    # x_train_data_org_sampled=x_train_data.sample(n=validation_data_new_moified_original.shape[0]).copy()
    db = tiny_db_connection(config["db_path"])
    try:
        db.drop_tables()
        adult_predictor = AdultPredictor()
        binary_counterfactual_obj = GenerateCounterfactualsBinary()
        validation_data_new_dd, numerical_features = \
            binary_counterfactual_obj.run(x_train_data=x_train_data,
                                          validation_data_new_moified=validation_data_new_moified,
                                          validation_data_new=validation_data_new,
                                          model=model, config=config,
                                          predictor=adult_predictor, y_val=y_val)
        validation_data_new_dd['latest_record_ind']='Y'
        validation_counterfactuals_collection = db.table(config["validation_generated_values_collection"])
        validation_counterfactuals_collection.insert_multiple(validation_data_new_dd.to_dict(orient="records"))
        if config["protected_attributes"] != [None]:
            binary_fairness_obj = GenerateFairnessBinary()
            fairness_score_dict, di_score_dict = binary_fairness_obj.run(config=config,
                                                                         numerical_features=numerical_features,
                                                                         db=db)
            validation_fairness_collection = db.table(config["fairness_score_collection"])
            validation_fairness_collection.insert(
                {**fairness_score_dict, **di_score_dict,"latest_record_ind":"Y"})
            logging.info("Fairness scores generated")
        # #
        binary_feature_attribution_obj = GenerateCounterfactualFeatureAttributionBinary()
        feature_attribution_train_df, feature_attribution_aggregated_df, \
        feature_attribution_aggregated_validation_df = binary_feature_attribution_obj.run(
            config=config, train_data_sampled=x_train_data_org, model=model, predictor=adult_predictor,
            validation_data=validation_data_new_moified_original, numerical_features=numerical_features)
        feature_attribution_aggregated_df['latest_record_ind'] = 'Y'
        feature_attribution_aggregated_collection = db.table(
            config["feature_attribution_aggregated_values_collection"])
        feature_attribution_aggregated_collection.insert_multiple(
            feature_attribution_aggregated_df.to_dict(orient='records'))
        feature_attribution_aggregated_validation_df['latest_record_ind'] = 'Y'
        validation_feature_attribution_aggregated_collection = db.table(
            config["validation_feature_attribution_aggregated_values_collection"])
        validation_feature_attribution_aggregated_collection.insert_multiple(
            feature_attribution_aggregated_validation_df.to_dict(orient='records'))
        feature_attribution_train_df['latest_record_ind'] ='Y'
        train_feature_attribution_aggregated_collection = db.table(
            config["train_feature_attribution_aggregated_values_collection"])
        train_feature_attribution_aggregated_collection.insert_multiple(
            feature_attribution_train_df.to_dict(orient='records'))
        logging.info("Feature attribution scores generated")

        #binary_live_prediction_obj = GenerateLivePredictions()
        # live_list = binary_live_prediction_obj.run(config=config, live_data=live_data_new_moified,
        #                                            predictor=adult_predictor,
        #                                            model=model, problem_type=config["problem_type"])
        # live_predictions_collection = db.table(config["live_prediction_values_collection"])
        # live_predictions_collection.insert_multiple(live_list)

        binary_performance_obj = GenerateBinaryPerformanceScore()
        performance_dict = binary_performance_obj.run(config=config, y_val=y_val, predictor=adult_predictor,
                                                      val_data=validation_data_new_moified_original, model=model)
        performance_dict["latest_record_ind"]="Y"
        performance_collection = db.table(config["performance_generated_values_collection"])
        performance_collection.insert(performance_dict)
        logging.info("Performance Metrics Generated")

    except Exception as error_message:
        logging.error(str(error_message))
        db.drop_tables()
        raise error_message
