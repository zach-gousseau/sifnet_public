# SIFNET
This repository contains code to accompany the paper "Probabilistic Gridded Seasonal Sea Ice Presence Forecasting using Sequence to Sequence Learning" by Asadi, N., Lamontagne, P., King, M., Richard, M., and Scott, K.A.

## Overview
### Data
Datasets are stored in the `datasets` folder. The `datasets/raw/` folder contains NetCDF4 files, e.g. as downloaded from ECMWF's CDS service. The `datasets/pre_computed/` folder contains serialized Numpy arrays created once the data pre-processing scripts have been run once (subsequent runs use these files rather than the raw data). Finally, the `datasets/yaml/` folder contains user-specified information on the spatial extent of interest, the training and test years, as well as computed aggregates on the dataset to improve efficiency. 

We've made pre-processed Numpy arrays for the Hudson Bay available [here](https://drive.google.com/drive/folders/1osYqBfz7VxNt-1aHEx_caXBKCe_5uhGa?usp=sharing) to allow users to by-pass downloading and processing ERA5 data. Download and untar the archive into `datasets/pre_computed/` to use them.

### SIFNET module
The `sifnet` folder is the main module which contains the experimental set-up and training procedures. This includes various utilities which are too numerous to describe here, but the files are relatively well-documented for interested users. Of note, the `model.py` file contains the Keras models tested in this project, and the `ice_presence_experiment.py` contains the main experiment class to create and train a model. 

### Running the training pipeline 
The full training procedure as described in the paper is written in `run_experiment.py`. This file can be invoked as a command line utility, and an example command to train a model for September can be found in `run_experiment.sh`. 

## Project Structure
```
.
├── requirements.txt    <-- Required Python packages
├── run_experiment.py   <-- Main file which runs the training procedure
├── run_experiment.sh   <-- Example command calling run_experiment.py
│
├── datasets
│   ├── raw/ERA5/*.nc         <-- Folder containing 'raw' NetCDF4 files covering the region of interest
│   ├── pre_computed/*.npy    <-- Folder containing pre-processed numpy arrays 
│   └── yaml/*.yaml           <-- Folder containing YAML files which specify spatial / temporal extents & store dataset information
│
└── sifnet
    ├── experiment.py  <-- Conatins case class for running experiments (must be inherited)
    │
    ├── data                            
    │   ├── data_manager                ┐ 
    │   │   ├── create_datasets.py      │  
    │   │   ├── dataset_functions.py    │-  Utilities for pre-processing & handling datasets, including the DatasetManager class
    │   ├── DatasetManager.py           │
    │   └──GeneratorFunctions.py        ┘
    │
    ├── medium_term_ice_forecasting
    │   ├── ice_presence
    │   │   ├── ice_presence_evaluation.py        <-- Model evaluation functions
    │   │   ├── ice_presence_experiment.py        <-- Contains main class for running an ice presence experiment
    │   │   ├── model.py                          <-- All Keras models
    │   ├── utilities                             
    │   │   ├── model_utilities.py                ┐
    │   │   ├── numpy_metrics.py                  │
    │   │   ├── postprocessing_tools.py           │
    │   │   ├── standardized_evaluations.py       │-  Various utilities used to run experiments  
    │   │   ├── reliability_diagram_vsklearn.py   │
    │   │   ├── visualization.py                  │
    │   │   └── visualize_correlation.py          ┘
    │   └── support_files
    │       └── era5-landsea-mask.nc              <-- Land mask for the ERA5 grid
    │
    └── utilities
        ├── gpu.py                  <-- Functions for training on GPU
        └── training_procedure.py   <-- Functions for training a single model
```
