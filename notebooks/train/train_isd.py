# Modified from https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/

import argparse
import os

# importing necessary libraries
import numpy as np

from sklearn import datasets
from azureml.core import Run,Dataset, Workspace,Datastore
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib

from azureml.core.run import Run
run = Run.get_context()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of estimator')
    parser.add_argument('--max_features', type=str, default='sqrt',
                        help='Max Feature')
    parser.add_argument('--dataset_name', type=str, default='ISDWeatherDS',
                        help='Name of the dataset')

    args = parser.parse_args()
    run.log('max_features', np.str(args.max_features))
    run.log('n_estimators', np.float(args.n_estimators))

    # loading the iris dataset
#     iris = datasets.load_iris()
    dataset = Dataset.get_by_name(run.experiment.workspace, args.dataset_name)  # NOQA: E402, E501
    dataset = dataset.to_pandas_dataframe()
    dataset = dataset.fillna(0)

    # X -> features, y -> label
    cols = list(dataset.columns)
    cols = [col for col in cols if dataset.dtypes[col] != 'object' and col not in ['version', 'datetime']]
    X = dataset[[col for col in cols if col not in ['temperature']]]
    y = dataset.temperature

    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


    rf = RandomForestRegressor(n_estimators=args.n_estimators, max_features=args.max_features).fit(X_train, y_train)
    rf_predictions = rf.predict(X_test)
    mse = mean_squared_error(y_test, rf_predictions)                    
    print('MSE  on test set: {:.2f}'.format(mse))

    # model accuracy for X_test
    run.log('MSE', np.float(mse))


    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(rf, 'outputs/model.joblib')


if __name__ == '__main__':
    main()
