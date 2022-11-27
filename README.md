# PCE prediction of OPV materials using Machine Learning 

## Introduction

This repository contains a regression model based on the two-layer feedforward artificial neural network for predicting the power conversion efficiency (PCE). This work demonstrates the possibility to train a neural network using descriptors for predicting photovoltaic properties.

## Model

The neural network model consists of two fully-connected layers with ReLU activations and a regression head. The dimensionality of the layers is shown below:

<img src="https://user-images.githubusercontent.com/4588093/72859687-d3ca9580-3d18-11ea-8f28-ff0e89d2940f.png" width="200">

As a loss function we use the mean squared error. The model has been trained using Adam optimizer.

## Results of evaluation

The squared correlation coefficient for the test set equals 0.83. The example of usage and the evaluation of the trained model is implemented in the script `model_test.py`. To run this script, one has to install all requirements by, for instance, invoking the command: `pip install -r requirements.txt`.



