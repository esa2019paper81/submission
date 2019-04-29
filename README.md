# Purpose

This repository contains the main implementation of PUFFINN: Universal and Fast FInding of Nearest Neighbors. The source code is provided in `PUFFINN`. It has been tested under different version of Ubuntu Linux.

## Building the Code

Follow the following steps in the `PUFFINN` directory. 

```
mkdir bin
cd bin && cmake ../
make
```

This creates a test binary, an example for how to use PUFFINN from the CPP side (see PUFFINN/examples/glove.cpp), and a Python wrapper for PUFFINN (see PUFFINN/python/example/example.py). Juse copy the `_puffinwrapper*` file to the directory where you want to run PUFFINN from. 

## Replication of Experiments in Paper

1. Get ann-benchmarks from https://github.com/erikbern/ann-benchmarks
2. Copy `algos-puffinn-evaluation.yaml` into the main ann-benchmarks directory.
3. Copy `Dockerfile.puffinn` into `ann_benchmarks/install/`.
4. Install `ann-benchmarks` following the guide on https://github.com/erikbern/ann-benchmarks
5. Run `python run.py --dataset glove-100-angular --definitions algos-puffinn-evaluation.yaml` from the main ann-benchmarks directory to run all experiments on Glove-1M. (Note that this will take several days. Modify the list of parameters used by PUFFINN to shorten the time it takes to carry out the experiment.)

## Looking at the Evaluation

All raw experimental results are found in `evaluation/all.csv`. Data and plots were evaluated using Jupyter notebook. The notebook is available at `evaluation/log.ipynb`. 
