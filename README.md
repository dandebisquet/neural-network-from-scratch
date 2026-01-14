# Neural Network From Scratch (Python)

A from-the-ground-up implementation of a simple neural network in **Python**, built to understand how neural networks work internally (without relying on high-level ML frameworks).

This project focuses on the core mechanics:
- forward propagation
- loss calculation
- backpropagation (gradients)
- parameter updates (training loop)

## Goals
- Build intuition for how neural networks learn
- Implement the training pipeline end-to-end
- Keep the code readable and educational

## What’s included
Depending on the version you upload, this project typically includes:
- A minimal neural network implementation (layers, weights, biases)
- Activation functions (e.g., sigmoid / tanh / ReLU)
- A loss function (commonly MSE or cross-entropy)
- Gradient-based optimization (e.g., gradient descent)
- A small demo/training task (toy dataset or simple classification/regression)

## How it works (high level)
1. **Initialize parameters** (weights + biases)
2. **Forward pass:** compute predictions from inputs
3. **Compute loss:** compare predictions to targets
4. **Backward pass:** compute gradients of loss w.r.t. parameters
5. **Update parameters:** move weights/biases in the direction that reduces loss
6. Repeat for multiple epochs until performance improves

## Requirements
- Python 3.9+ recommended

If you used third-party packages (common: NumPy, Matplotlib), install with:
```bash
pip install -r requirements.txt
