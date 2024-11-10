# CNN from scratch

This is just a simple deep learning "framework" that implements some layers from scratch using only numpy as a dependency. The goal was to learn the math inside a very simple neural network architecture with conv layers, fully connected layers, ReLU, etc. I hope that somebody can find this useful.

It also includes a numba `njit` decorator to improve CPU performance by a lot (keep reading to find out by how much).

## How to Run 🏃

To make a simple demonstration i've included a script that runs a network composed of two conv layers and a softmax classifier for the MNIST dataset.

1. create virtual env and install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. run

```bash
python main.py
```

## Implementation details 📚

### Layers implemented so far 🎂

- Convolutional
- Dense
- Max Pooling
- ReLU
- Softmax classifier

### MNIST CNN example 🔬🧪

To my further understanding of convolutional neural networks i coded a simple architecture to train on the MNIST dataset.
The architecture is a simple two layer cnn layers with a dense layer at the end. This image may clarify it enough for the readers:

![arch](/docs/arch.png "architecture")

#### results:

- training set size: 1000 images
- test set size: 10 images
- epochs: 20

```
processing data...
data processed succesfully
training network...
Processing |################################| 1000/1000
Epoch 1 Loss= 0.163019 Run in: 21.16 seconds
Processing |################################| 1000/1000
Epoch 2 Loss= 0.064415 Run in: 21.22 seconds
Processing |################################| 1000/1000
Epoch 3 Loss= 0.046571 Run in: 21.55 seconds
Processing |################################| 1000/1000
Epoch 4 Loss= 0.039135 Run in: 21.49 seconds
Processing |################################| 1000/1000
Epoch 5 Loss= 0.034347 Run in: 21.3 seconds
Processing |################################| 1000/1000
Epoch 6 Loss= 0.030685 Run in: 21.54 seconds
Processing |################################| 1000/1000
Epoch 7 Loss= 0.027669 Run in: 21.12 seconds
Processing |################################| 1000/1000
Epoch 8 Loss= 0.025164 Run in: 21.3 seconds
Processing |################################| 1000/1000
Epoch 9 Loss= 0.023078 Run in: 21.3 seconds
Processing |################################| 1000/1000
Epoch 10 Loss= 0.021287 Run in: 21.08 seconds
Processing |################################| 1000/1000
Epoch 11 Loss= 0.019683 Run in: 21.05 seconds
Processing |################################| 1000/1000
Epoch 12 Loss= 0.018196 Run in: 21.12 seconds
Processing |################################| 1000/1000
Epoch 13 Loss= 0.016791 Run in: 21.07 seconds
Processing |################################| 1000/1000
Epoch 14 Loss= 0.015449 Run in: 21.18 seconds
Processing |################################| 1000/1000
Epoch 15 Loss= 0.014158 Run in: 21.21 seconds
Processing |################################| 1000/1000
Epoch 16 Loss= 0.012906 Run in: 21.11 seconds
Processing |################################| 1000/1000
Epoch 17 Loss= 0.011677 Run in: 21.29 seconds
Processing |################################| 1000/1000
Epoch 18 Loss= 0.010469 Run in: 21.32 seconds
Processing |################################| 1000/1000
Epoch 19 Loss= 0.009296 Run in: 21.3 seconds
Processing |################################| 1000/1000
Epoch 20 Loss= 0.008183 Run in: 21.33 seconds
network trained
pred: 7 	true: 7
pred: 6 	true: 2
pred: 1 	true: 1
pred: 0 	true: 0
pred: 4 	true: 4
pred: 1 	true: 1
pred: 4 	true: 4
pred: 9 	true: 9
pred: 2 	true: 5
pred: 9 	true: 9

 >>> Program run in 428.72 seconds <<<

```

Now by using installing the numba package and decorating the main matrix multiplication loop function we can get a free performance upgrade.

Running the same test we get:

```
processing data...
data processed succesfully
training network...
Processing |################################| 1000/1000
Epoch 1 Loss= 0.226436 Run in: 4.41 seconds
Processing |################################| 1000/1000
Epoch 2 Loss= 0.191109 Run in: 3.0 seconds
Processing |################################| 1000/1000
Epoch 3 Loss= 0.095432 Run in: 3.08 seconds
Processing |################################| 1000/1000
Epoch 4 Loss= 0.057641 Run in: 3.29 seconds
Processing |################################| 1000/1000
Epoch 5 Loss= 0.045776 Run in: 3.5 seconds
Processing |################################| 1000/1000
Epoch 6 Loss= 0.039426 Run in: 3.4 seconds
Processing |################################| 1000/1000
Epoch 7 Loss= 0.035062 Run in: 2.92 seconds
Processing |################################| 1000/1000
Epoch 8 Loss= 0.03169 Run in: 2.85 seconds
Processing |################################| 1000/1000
Epoch 9 Loss= 0.028919 Run in: 2.8 seconds
Processing |################################| 1000/1000
Epoch 10 Loss= 0.026561 Run in: 2.81 seconds
Processing |################################| 1000/1000
Epoch 11 Loss= 0.024518 Run in: 3.0 seconds
Processing |################################| 1000/1000
Epoch 12 Loss= 0.022723 Run in: 3.33 seconds
Processing |################################| 1000/1000
Epoch 13 Loss= 0.021129 Run in: 3.03 seconds
Processing |################################| 1000/1000
Epoch 14 Loss= 0.019699 Run in: 2.94 seconds
Processing |################################| 1000/1000
Epoch 15 Loss= 0.018398 Run in: 3.0 seconds
Processing |################################| 1000/1000
Epoch 16 Loss= 0.0172 Run in: 3.11 seconds
Processing |################################| 1000/1000
Epoch 17 Loss= 0.016086 Run in: 2.87 seconds
Processing |################################| 1000/1000
Epoch 18 Loss= 0.015042 Run in: 4.09 seconds
Processing |################################| 1000/1000
Epoch 19 Loss= 0.014057 Run in: 4.19 seconds
Processing |################################| 1000/1000
Epoch 20 Loss= 0.013128 Run in: 3.67 seconds
network trained
pred: 7 	true: 7
pred: 6 	true: 2
pred: 1 	true: 1
pred: 0 	true: 0
pred: 4 	true: 4
pred: 1 	true: 1
pred: 4 	true: 4
pred: 9 	true: 9
pred: 2 	true: 5
pred: 9 	true: 9

 >>> Program run in 69.68 seconds <<<
```

We see a wooping ~80% improvement.

Somehow adding the option `parallel=True` made the performance worst, maybe because of the unnecesary overhead to make it run on each thread.

## TODO 📋

- [x] conv net
- [x] maxpool layer
- [x] relu
- [x] fully connected layer
- [x] softmax classifier
- [x] download minst
- [x] data loader
- [x] better params initialization (Xavier)
- [x] train basic classifier on dataset
- [x] better error measurement
- [ ] plot error vs epochs
- [ ] input args from console for the main script

## maybe in the future

- [ ] Batch Norm
- [ ] adam
- [ ] dropdown layer
- [ ] resnet
