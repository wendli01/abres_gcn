# Reconciliation of Mental Concepts with Graph Neural Networks

This repository accompanies the DEXA 2022 paper [Reconciliation of Mental Concepts with Graph Neural Networks]().

It contains all code as well as experimental setups described in the paper including results with al visualizations as standalone `jupyter` notebooks.


If you use code, data or any results in this repository, please cite:
>

## Experiments

Complete experiments are stored in the notebooks for [link prediction](link_prediction.py) and basic [network analysis](network_analysis.ipynb).

## Dataset

The [TEAM-IP-1 Dataset](team_ip_1.zip)  described in the paper is also included in this repository.

## Installation


Installation via the provided conda envirionment is encouraged.

> `conda env create -f abres_gcn.yml`


To replicate the experiments, [`jupyter`](https://jupyter.org/install) needs to be installed as well, e.g. with


> `conda install -c conda-forge notebook`
> 
> or 
> 
> `pip install jupyterlab`


## Usage


All models and transformers are implemented as `sklearn` Estimators.


```python
from link_prediction import LinkPredictor
import networkx as nx

# training graph
X_train: nx.DiGraph
# test graph to indicate potential edges
X_test : nx.DiGraph

abres_gcn = LinkPredictor()
abres_gcn.fit(X_train)
predictions = abres_gcn.predict(X_train, X_test)
```
