# ASTGI
# Rethinking Irregular Time Series Forecasting:  A Simple yet Effective Baseline

## Introduction

<div align="center">
<img alt="Logo" src="figs/overview.png" width="80%"/>
</div>

Overview of the ASTGI framework. (a) Directly representing each discrete observation as a spatio-temporal point. (b) Adaptively constructing a causal graph for each point. (c) Iteratively propagating information on the adaptive graphs to update features. (d) Unifying prediction as a neighborhood aggregation task for a query point.

## Quickstart
> [!IMPORTANT]
> this project is fully tested under python 3.11, it is recommended that you set the Python version to 3.11 and CUDA version to 12.0
1. Installation:

> ```shell
> pip install -r requirements.txt
> ```

2. Data preparation:
    1. Follow the processing scripts in [gru_ode_bayes](https://github.com/edebrouwer/gru_ode_bayes/tree/master/data_preproc/MIMIC) to get complete_tensor.csv.
    2. Put the result under ~/.tsdm/rawdata/MIMIC_III_DeBrouwer2019/complete_tensor.csv.
3. Train and evaluate model:

We provide the experiment scripts for all benchmarks under the folder `./APN/scripts`. For example you can reproduce a experiment result as the following:

```shell
bash ./scripts/USHCN.sh
```

