# CNN from scratch

This is just a simple deep learning "framework" that implements some layers from scratch using only numpy as a dependency.

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

  - forward

    $Z^{[l]}_{h,w,c} = \sum_{i=0}^{k-1} \sum_{j=0}^{k-1} \sum_{c'} W^{[l]}_{i,j,c',c} \cdot A^{[l-1]}_{sh+i,sw+j,c'} + b^{[l]}_c$

  - backward

    $\frac{\partial L}{\partial W^{[l]}} = \sum_{h,w} \frac{\partial L}{\partial Z^{[l]}_{h,w}} \cdot A^{[l-1]}_{sh+i,sw+j}$

    $\frac{\partial L}{\partial b^{[l]}} = \sum_{h,w} \frac{\partial L}{\partial Z^{[l]}_{h,w}}$

    $\frac{\partial L}{\partial A^{[l-1]}} = \sum_{h,w} W^{[l]} * \frac{\partial L}{\partial Z^{[l]}_{h,w}}$

- Dense
  - forward
  - backward
- Max Pooling

  - forward

    $P^{[l]}_{h,w,c} = \max_{i,j \in k×k} A^{[l-1]}_{sh+i,sw+j,c}$

  - backward

    $\frac{\partial L}{\partial A^{[l-1]}_{i,j,c}} = \frac{\partial L}{\partial P^{[l]}_{h,w,c}} \cdot \mathbb{1}(A^{[l-1]}_{i,j,c} = \max_{i',j' \in k×k} A^{[l-1]}_{i',j',c})$

- ReLU

  - forward

    $A^{[l]} = \max(0, Z^{[l]})$

  - backward

    $\frac{\partial L}{\partial Z^{[l]}} = \frac{\partial L}{\partial A^{[l]}} ⊙ \mathbb{1}(Z^{[l]} > 0)$

- Softmax classifier

  - loss

    $P(y=j|Z^{[L]}) = \frac{e^{Z^{[L]}_j}}{\sum_{k=1}^K e^{Z^{[L]}_k}}$

    $L = -\sum_{j=1}^K y_j \log(P(y=j|Z^{[L]}))$

  - backward

    $\frac{\partial L}{\partial Z^{[L]}_i} = P(y=i|Z^{[L]}) - y_i$

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

- [ ] dropdown layer
- [ ] resnet