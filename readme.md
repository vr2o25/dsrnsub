# Deep Sets of Random Neighbors with Attention


## Repo structure

```
|-> datasets 
    |-> synthetic
    |-> taxibj
|-> data_exploration
    |-> visualizing_sampling_strategies.py
|-> baseline_knn
|-> ours_DSRN
    |-> synthetic
    |-> taxibj
```

## Data exploration


### `visualizing_sampling_strategies.py`


Arguments `--dataset fixed` (default) or `--dataset flight`


Plots the spatial position of the data (test partition) captured by the sensors, 
with most recent data points in a more opaque tone, and older data points with a more transparent tone.


The argument allows to visualize the data captured for the two sampling strategies used: fixed sensors and moving sensors.


### `dataset_histograms.py`


`--dataset synthetic` (default) or `--dataset taxibj`: This argument changes the dataset being explored. 


This scripts plots the histograms of the two datasets used, the synthetic and TaxiBJ datasets.


## Baseline KNN


### `evaluate.py`


This script is used to evaluate the baseline KNN with bootstrapping method.


Arguments:


`--type fixed` (default) or `--type flight`: This argument changes the sampling strategy used.


`--dataset synthetic` (default) or `--dataset taxibj`: This argument changes the dataset being evaluated. 


## Ours - DSRN


Folders synthetic and taxibj, with the scripts used for the two different datasets.


Each folder has the following files:


### `dataset.py`


Dataset class for the corresponding dataset, with all the necessary transformation and data normalization and augmentation settings.


### `model3.py` and `model3c.py`


Model architecture, in two slighly different variants, changing the position of the batch normalization layer.


### `training.py` and `evaluate.py`


Script for training and evaluation of the model.
