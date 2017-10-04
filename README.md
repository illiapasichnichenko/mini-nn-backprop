# mini-nn-backprop
## Overview
The main purpose of this code is to illustrate the backpropagation algorithm. In this problem, a small neural network learns how to decide whether a numeric vector belongs to a certain region. Concretely, inputs are vectors with coordinates between -2 and 2, and the region is the unit sphere. The net consists of an input layer, a hidden layer with the sigmoid activation, and an output layer with the softmax activation. It is described by the following equations

![equation](http://latex.codecogs.com/gif.latex?z%5E1%20%3D%20W%5E1x%20&plus;%20b%5E1%2C%5Cquad%20h%20%3D%20S%28z%5E1%29%2C%5Cquad%20z%5E2%20%3D%20W%5E2h%20&plus;%20b%5E2%2C%5Cquad%20y%20%3D%20%5Csigma%28z%29%2C)

where x is an input vector, y is an output probability vector, W1, W2, b1, b2 are model parameters.  

## Usage

To run the algorithm, use the following command `python main.py`. You need only numpy for this.

The repo also contains equivalent code for tensorflow, use `python main_tf.py` to run.
