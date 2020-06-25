# EITEST
An implementation of EITEST (Event Information TEST) to discover statistical relationships between event series and time series. The methodology is described and analysed in detail in the paper:

Erik Scharwächter, Emmanuel Müller: **Two-Sample Testing for Event Impacts in Time Series.**
In: Proceedings of the 2020 SIAM International Conference on Data Mining (SIAM SDM)
[[permalink]](https://doi.org/10.1137/1.9781611976236.2)
[[arXiv preprint]](https://arxiv.org/abs/2001.11930)

## Contact and Citation

* Corresponding author: [Erik Scharwächter](mailto:scharwaechter@bit-uni-bonn.de)
* Please cite our paper if you use or modify our code for your own work.

## Requirements and Usage

Our module [eitest.py](./eitest.py) was developed with [Python 3.7](https://www.python.org/) and requires [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) and [Numba](https://numba.pydata.org/). Check the Jupyter notebook [ampds.ipynb](./ampds.ipynb) to reproduce the results from our paper and understand how to use our module. The notebook additionally requires [Pandas](https://pandas.pydata.org/).

## License

The code is released under the MIT license.
