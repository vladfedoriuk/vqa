"""
The pipelines defined for the datasets.

The pipelines leverage the DVC framework to track the datasets.

https://dvc.org/doc/start/data-management/data-pipelines

Currently, the supported pipelines are: DAQUAR.

The DAQUAR pipeline is used to process the DAQUAR dataset:

- Loading the dataset.
- Flattening the comma-separated answers.
- Saving the datasets as CSV files.

The pipeline is defined in the pipelines/process_daquar.py module.

The DAQUAR dataset is tracked with DVC.

If you don't have it locally, you can download it with:

dvc pull

The DAQUAR dataset is stored in the data/raw/daquar directory.

The processed DAQUAR dataset is stored in the data/processed/daquar directory.

The processed DAQUAR dataset is used by the models.
"""
