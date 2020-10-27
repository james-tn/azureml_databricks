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
from azureml.pipeline.steps import HyperDriveStep, HyperDriveStepRun

from azureml.core.run import Run
run = Run.get_context()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='ISDWeatherDS',
                        help='Name of the dataset')
    parser.add_argument('--hd_step_name', type=str, default='hyper_param_training',
                        help='hyper_param_training')
    args = parser.parse_args()

    dataset = Dataset.get_by_name(run.experiment.workspace, args.dataset_name)  # NOQA: E402, E501
    dataset = dataset.to_pandas_dataframe()
    dataset = dataset.fillna(0)

    cols = list(dataset.columns)
    cols = [col for col in cols if dataset.dtypes[col] != 'object' and col not in ['version', 'datetime']]
    X = dataset[[col for col in cols if col not in ['temperature']]]
    y = dataset.temperature
    
    #get parent run id
    parent_run = run.parent
    hd_step_name = args.hd_step_name
    hd_step_run = HyperDriveStepRun(step_run=parent_run.find_step_run(hd_step_name)[0])
    best_run = hd_step_run.get_best_run_by_primary_metric()
    best_run.download_file('outputs/model.joblib')
    model_file ='model.joblib'
    model = joblib.load(model_file)

    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)                    
    print('MSE  on test set: {:.2f}'.format(mse))
    run.log('MSE', np.float(mse))
    model_name = "isd_model"
    metrics ={}
    current_mse=None
    try:
        current_model = Model(name=model_name, workspace=ws)

        current_mse = current_model.tags["mse"]
    except:
        print("Exception getting model or getting tag")
    metrics["mse"] = mse


    if (current_mse):
        if (current_mse >=mse):
            print("better model found, registering new model")
            run.upload_file(model_name,model_file)
            run.register_model(
                           model_path = model_name,
                           model_name = model_name,
                           tags=metrics)
    else:
        print("first time registering model")
        run.upload_folder(model_name,model_file)
        run.register_model(
                       model_path = model_name,
                       model_name = model_name,
                       tags=metrics)



if __name__ == '__main__':
    main()
