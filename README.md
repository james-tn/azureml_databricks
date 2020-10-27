# Training examples on Azure ML & Databricks Integration and ML Model Interpretability 
This is a training material on follow topics:
1. Enable querying DeltaLake data from Azure ML with Dask cluster
2. Model Interpretability with azureml-interpret package 
3. Distributed Hyper-Param tunning with AzureML Hyperopt.
The Databricks folder contains a databricks notebook to setup Databricks so that it output manifest files whenver there's a change in the Deltalake table data. In addition, there's 2nd notebook to demostrate how to write Databricks's transformation to AML Dataset.
The Notebook folder contains examples to run in Azure ML Compute Instance environment. The notebooks are:
      - setup_dask.ipynb: setting up dask cluster
      - analyze_deltalake_with_dask.ipynb: analyze deltalake data from Azure ML Dask
      - train-hyperparameter-tune.ipynb: example of how to tune ML model hyper parameters in a multinodes AML cluster 
      - pipeline_definition.ipynb: example of how to have hyper param tunning + model testing/registration in a pipeline
