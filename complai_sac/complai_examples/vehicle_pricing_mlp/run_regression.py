import logging
from extendables.get_predictions import GetPredictions
from generate_values.generate_counterfactuals_regression import GenerateCounterfactualsRegressor
from generate_values.generate_drift_score_counterfactual_regression import \
    GenerateCounterfactualFeatureAttributionRegression
from generate_values.generate_performance_values_regression import GenerateRegressionPerformanceScore
from generate_values.generate_fairness_regression import GenerateFairnessRegression
from generate_values.generate_predictions_live import GenerateLivePredictions
from utils.get_data_generic import GenericGetData
from utils.get_connections import tiny_db_connection


class HousingPriceRegressor(GetPredictions):
    def __init__(self):
        super().__init__()

    def get_scores(self, X_data=None, model=None):
        regression_value = model.predict(X_data)
        return regression_value


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)
    yaml_path = "./vehicle_pricing_mlp_config.yml"
    data_obj = GenericGetData(yaml_path=yaml_path)
    x_train_data, validation_data_new_moified, validation_data_new, live_data_new_moified, y_val, model, config \
        , x_train_data_org = \
        data_obj.get_data()
    # print(pd.DataFrame(x_train_data))
    validation_data_new_moified_original = validation_data_new_moified.copy()
    # x_train_data_org_sampled=x_train_data.sample(n=validation_data_new_moified_original.shape[0]).copy()
    db = tiny_db_connection(config["db_path"])
    try:
        db.drop_tables()
        housing_predictor = HousingPriceRegressor()
        regressor_counterfactual_obj = GenerateCounterfactualsRegressor()
        validation_data_new_dd, numerical_features = regressor_counterfactual_obj.run(x_train_data=x_train_data,
                                                                                      validation_data_new_moified=validation_data_new_moified,
                                                                                      validation_data_new=validation_data_new,
                                                                                      model=model,
                                                                                      config=config,
                                                                                      predictor=housing_predictor,
                                                                                      y_val=y_val)
        validation_data_new_dd['latest_record_ind'] = 'Y'
        validation_counterfactuals_collection = db.table(config["validation_generated_values_collection"])
        validation_counterfactuals_collection.insert_multiple(validation_data_new_dd.to_dict(orient="records"))

        if config["protected_attributes"] != [None]:
            regressor_fairness_obj = GenerateFairnessRegression()
            fairness_score_dict, di_score_dict = regressor_fairness_obj.run(config=config,
                                                                            numerical_features=numerical_features,
                                                                            db=db)
            validation_fairness_collection = db.table(config["fairness_score_collection"])
            validation_fairness_collection.insert(
                {**fairness_score_dict, **di_score_dict,"latest_record_ind":"Y"})
            logging.info("Fairness scores generated")

        regressor_feature_attribution_obj = GenerateCounterfactualFeatureAttributionRegression()

        feature_attribution_train_df, feature_attribution_aggregated_df, feature_attribution_aggregated_validation_df \
            = regressor_feature_attribution_obj.run(
            train_data_sampled=x_train_data_org, validation_data=validation_data_new_moified_original,
            model=model,
            config=config, numerical_features=numerical_features,
            predictor=housing_predictor)
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
        feature_attribution_train_df['latest_record_ind'] = 'Y'
        train_feature_attribution_aggregated_collection = db.table(
            config["train_feature_attribution_aggregated_values_collection"])
        train_feature_attribution_aggregated_collection.insert_multiple(
            feature_attribution_train_df.to_dict(orient='records'))
        logging.info("Feature attribution scores generated")

        # regressor_live_prediction_obj = GenerateLivePredictions()
        # regressor_live_prediction_obj.run(config=config, live_data=live_data_new_moified, predictor=housing_predictor,
        #                                   model=model, problem_type=config["problem_type"])
        regressor_performance_obj = GenerateRegressionPerformanceScore()
        performance_dict = regressor_performance_obj.run(config=config, y_val=y_val, predictor=housing_predictor,
                                                         val_data=validation_data_new_moified_original, model=model)
        performance_dict["latest_record_ind"] = "Y"
        performance_collection = db.table(config["performance_generated_values_collection"])
        performance_collection.insert(performance_dict)
        logging.info("Performance Metrics Generated")

    except Exception as error_message:
        logging.error(str(error_message))
        db.drop_tables()
        raise error_message
