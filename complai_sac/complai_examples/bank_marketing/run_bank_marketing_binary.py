import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import mlflow
import logging
from urllib.parse import urlparse
import yaml
from extendables.get_predictions import GetPredictions
from utils.get_connections import tiny_db_connection
from complai_utils.complai_functions import run_complai_scan_binary, get_trust_scores

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


class BinaryPredictor(GetPredictions):
    def __init__(self):
        super().__init__()

    def get_proba_scores(self, X_data=None, model=None):
        probability_scores = model.predict_proba(X_data)
        return probability_scores[:, 1]


def get_dummy_from_bool(row, column_name):
    ''' Returns 0 if value in column_name is no, returns 1 if value in column_name is yes'''
    return 1 if row[column_name] == 'yes' else 0


def get_correct_values(row, column_name, threshold, df):
    ''' Returns mean value if value in column_name is above threshold'''
    if row[column_name] <= threshold:
        return row[column_name]
    else:
        mean = df[df[column_name] <= threshold][column_name].mean()
        return mean


def clean_data(df):
    '''
    INPUT
    df - pandas dataframe containing bank marketing campaign dataset

    OUTPUT
    df - cleaned dataset:
    1. columns with 'yes' and 'no' values are converted into boolean variables;
    2. categorical columns are converted into dummy variables;
    3. drop irrelevant columns.
    4. impute incorrect values
    '''

    cleaned_df = df.copy()

    # convert columns containing 'yes' and 'no' values to boolean variables and drop original columns
    # bool_columns = ['default', 'housing', 'loan', 'deposit']
    bool_columns = ['deposit']
    for bool_col in bool_columns:
        cleaned_df[bool_col + '_bool'] = df.apply(lambda row: get_dummy_from_bool(row, bool_col), axis=1)

    cleaned_df = cleaned_df.drop(columns=bool_columns)

    # convert marital column to numerical values
    my_dict = {'married': 1, 'single': 2, 'divorced': 3}
    cleaned_df = cleaned_df.replace({"marital": my_dict})

    # convert categorical columns to dummies
    # cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

    # for col in  cat_columns:
    #    cleaned_df = pd.concat([cleaned_df.drop(col, axis=1),
    #                            pd.get_dummies(cleaned_df[col], prefix=col, prefix_sep='_',
    #                                           drop_first=True, dummy_na=False)], axis=1)

    # drop irrelevant columns
    cleaned_df = cleaned_df.drop(columns=['pdays'])

    # impute incorrect values and drop original columns
    cleaned_df['campaign_cleaned'] = df.apply(lambda row: get_correct_values(row, 'campaign', 34, cleaned_df), axis=1)
    cleaned_df['previous_cleaned'] = df.apply(lambda row: get_correct_values(row, 'previous', 34, cleaned_df), axis=1)

    cleaned_df = cleaned_df.drop(columns=['campaign', 'previous'])

    return cleaned_df


if __name__ == "__main__":
    try:
        with open('./bank_marketing_xgb_config.yml', 'r') as fl:
            params = yaml.safe_load(fl)

        mlflow.set_experiment(params["experiment"])
        with mlflow.start_run() as run:
            df = pd.read_csv('./data/bank.csv')
            cleaned_df = clean_data(df)
            cleaned_df.head()

            X = cleaned_df.drop(columns='deposit_bool')
            y = cleaned_df[['deposit_bool']]

            target_y = y['deposit_bool']

            X_train, X_test, y_train, y_test = train_test_split(X, target_y, test_size=params["test_size"],
                                                                random_state=params["rand_state"])

            cat_feat = params["categorical_features_indexes"]
            num_feat = [i for i in range(len(X_train.columns)) if i not in cat_feat]

            clf = Pipeline([
                ('PP', ColumnTransformer([
                    ('num', StandardScaler(), num_feat),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feat)])),
                # ('SVC',SVC(gamma='auto',probability=True))])
                # ('LR',LogisticRegression())])
                ('xgb', XGBClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"],
                                      gamma=params["gamma"], subsample=params["subsample"],
                                      colsample_bytree=params["colsample_bytree"], max_depth=params["max_depth"]))])

            clf.fit(X_train.values, y_train.values)
            y_train_preds = clf.predict(X_train)
            y_test_preds = clf.predict(X_test)

            clf.fit(X_train.values, y_train.values)
            precision = precision_score(y_test.values, y_test_preds)
            recall = recall_score(y_test.values, y_test_preds)
            accuracy = clf.score(X_test.values, y_test.values)
            metrics_dict = {"accuracy": accuracy,"precision":precision,"recall":recall}
            db = tiny_db_connection(db_path=params["db_path"])
            db.drop_tables()
            X_test.reset_index(drop=True, inplace=True)
            X_train.reset_index(drop=True, inplace=True)
            y_test.reset_index(drop=True, inplace=True)
            x_train_data = X.values

            predictor = BinaryPredictor()

            run_complai_scan_binary(x_train_data=x_train_data, validation_data_new=X_test, model=clf,
                                    config=params,
                                    predictor=predictor, y_val=y_test, db=db, x_train_data_org=X_train)

            scores_dict = get_trust_scores(
                validation_generated_values_collection=params[
                    "validation_generated_values_collection"],
                fairness_score_collection=params['fairness_score_collection'],
                performance_generated_values_collection=params[
                    'performance_generated_values_collection'],
                feature_attribution_aggregated_values_collection=params[
                    'feature_attribution_aggregated_values_collection']
                , db=db, problem_type=params["problem_type"])

            metrics_dict_fin = {**metrics_dict, **scores_dict}
            logging.info(metrics_dict_fin)
            mlflow.log_metrics(metrics_dict_fin)
            mlflow.log_artifact(os.path.join("db.json"))
            mlflow.log_param('random_seed', params["rand_state"])
            mlflow.log_param('test_size', params["test_size"])
            mlflow.log_param('n_estimators', params["n_estimators"])
            mlflow.log_param('learning_rate', params["learning_rate"])
            mlflow.log_param('gamma', params["gamma"])
            mlflow.log_param('subsample', params["subsample"])
            mlflow.log_param('max_depth',params["max_depth"])

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(clf, "model", registered_model_name="XGBoostBankMarketing")
            else:
                mlflow.sklearn.log_model(clf, "model")

    except Exception as error_message:
        logging.error(str(error_message))
        raise error_message
    finally:
        logging.info("Run Completed")
