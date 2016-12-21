# K-FAC and various second-order optimizers in Theano

This project attempts to reproduce the results in the paper:
[Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://arxiv.org/pdf/1503.05671v6.pdf)

Currently, list of implemented algos consists of GD, AdaGrad, RMSProp, NAG, Adam, Gauss-Newton, Fisher, Khatri-Rao Fisher, block-diagonal Khatri-Rao Fisher. 

For now, K-FAC is assumed to be block-diagonal KR Fisher with rescaling and momentum turned on. Proper K-FAC implementation is in progress and will be added soon.

## How to run
### Software
* The code is written in Python 3
* `pip3 install theano scipy matplotlib`
* `pip3 install keras sklearn` (for datasets)
* `pip3 install scikit-cuda` (optionally)

### Basic usage
* To specify model, optimizer and other params you still need to modify them in the code
* `python3 main.py` will train the model and log some stats to a .csv file in `tests/` directory
* To view plots you can use provided `plt_adv.py` script like

        ./plt_adv.py loss,grad_mean tests/test_digits_1500_div_2_classify_16-15-15-10_*

