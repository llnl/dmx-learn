DMixLearn - (Distributed Mixture Learning) A package for distributed heterogeneous density estimation. With only a few lines of code you can specify and fit complex models on variable-length heterogenous data.

--------------------------------------------------------------------------------

## 📚 Documentation
View the full documentation on **Read the Docs**:

👉 [https://dmixlearn.readthedocs.io/en/latest/](https://dmixlearn.readthedocs.io/en/latest/)

## Installation

User installation with pip
```
> pip install --user /path/to/package
```


## Building with mpi4py and umap-learn

```
> cd /path/to/package
> pip install --user .[optional]
```

## Stats Examples
Examples using `stats` distributions that run locally are located in ./dml/examples/

```
> export PYHONPATH=$PYTHONPATH:/path./to/dmixlearn
> PYTHONPATH=/path/to/package/ python ./dml/examples/stats_examples/mixture_example.py
```

## Running with Spark
Examples that run with Apache Spark are located in./dml/examples_spark/

First build a wheel
```
> cd /path/to/dmixlearn
> pip install setuptools wheel
> python setup.py bdist_wheel
```

Run the example with below
```
> /path/to/spark/bin/spark-submit --master local[*] --py-files /path/to/package/dist/dmixlearn-0.1.8.4-py3-none-any.whl ./dml/examples_spark/mixture_example.py
```

## Running with MPI4PY
Examples that run with mpi4py are located in ./dml/mpi4py/examples/

Below will run the example ./dml/mpi4py/examples/estimation_example.py with 4 cores.
```
> export PYHONPATH=$PYTHONPATH:/path./to/dmixlearn
> mpiexec -n 4 python /path/to/package/dml/mpi4py/examples/estimation_example.py
```
