# SIFNET
This repository contains code to accompany the paper "Probabilistic Gridded Seasonal Sea Ice Presence Forecasting using Sequence to Sequence Learning" by Asadi, N., Lamontagne, P., King, M., Richard, M., and Scott, K.A.

## Project Structure
```
.
├── requirements.txt    <-- Required Python packages
├── run_experiment.py   <-- Main file which runs the training procedure
├── run_experiment.sh   <-- Example command calling run_experiment.py
│
├── datasets
│   ├── pre_computed/*.npy    <-- Folder containing pre-processed numpy arrays 
│   ├── raw/ERA5/*.nc         <-- Folder containing 'raw' NetCDF4 files covering the region of interest
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