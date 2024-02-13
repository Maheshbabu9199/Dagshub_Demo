# this file is to demonstrate the dvc 

from logger import logger
import os
import sys
import pandas as pd
import numpy as np
import typing 
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
import mlflow
import mlflow.sklearn, mlflow.data


mlflow.set_experiment('first_experiment_dvcdemo')


def get_data(filepath: Path) -> pd.DataFrame:
    """
    Returns the dataset as csv file
    """
    try:
        logger.debug('Reading data from {0}'.format(filepath))
        logger.debug('reading data %s'%filepath)
        data = pd.read_csv(filepath)
        logger.warning("Data loaded successfully")
        return data
    except Exception as e:
        #logger.error('error in reading data')
        logger.error("Error reading data from %s"%filepath)


def prepare_data(data: pd.DataFrame, sizetest: float) -> pd.DataFrame:

    features = data.drop(columns=['price'])
    label = data['price']

    
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=sizetest, random_state=0)

    
    cat_cols = data.select_dtypes(include='object')
    num_cols = data.select_dtypes(exclude='object')
    
    label = LabelEncoder()
    scaler = StandardScaler()

    for cat_col in cat_cols:
        X_train[cat_col] = label.fit_transform(X_train[cat_col])
        X_test[cat_col] = label.transform(X_test[cat_col])

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(data=X_train, columns=features.columns.tolist())
    X_test = pd.DataFrame(data=X_test, columns=features.columns.tolist())

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    try:
        with mlflow.start_run(run_name='artifacts_run'):

            logger.info('started main program')
            filepath = Path('data\Housing.csv')
            data = get_data(filepath)
            logger.warning('completed reading data')
            sizetest = 0.4
            mlflow.log_param('sizetest', sizetest)
            X_train, X_test, y_train, y_test = prepare_data(data, sizetest)

            temp_data = pd.concat([X_train, y_train], axis=1)

            temp_data.to_csv('temp_data.csv')
            temp = os.path.join(os.getcwd(), 'temp_data.csv')
            mlflow.log_artifact(Path(temp))
           

            linear = LinearRegression()

            linear.fit(X_train, y_train)

            training_dataset = mlflow.data.from_pandas(X_train) 
            mlflow.log_input(training_dataset, context='training')

            y_pred = linear.predict(X_test)

            mse_score = mse(y_pred, y_test)
            mae_score = mae(y_pred, y_test)

            metrics = {'mse_score': mse_score, 'mae_score': mae_score}
           
            mlflow.log_metrics(metrics)
           
            print('params:\nsizetest:- {0}'.format(sizetest))
            print('metrics:\nmetrics:- {0}'.format(metrics))

            mlflow.sklearn.log_model(sk_model=linear, artifact_path='models', registered_model_name='linear_model')
    except Exception as e:
        logger.critical('Failed to complete main program')
        raise e