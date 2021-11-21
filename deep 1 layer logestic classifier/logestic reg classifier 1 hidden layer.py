#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:14:22 2021

@author: lemo
"""

#Logestic regression classifier
import numpy as np
np.random.seed(1)


def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s


def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y
X , Y = load_planar_dataset()
#print(Y)

#initialze layer_size
def layer_size(X,Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x,n_h,n_y)
print(layer_size(X, Y))

def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(1)
    w1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    w2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    
    assert (w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (w2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameter = {
        "W1":w1,
        "W2":w2,
        "b1":b1,
        "b2":b2,
        }
    return parameter
#print(layer_size(X, Y)[0])
#parameters = initialize_parameters(2,4,1)
#print("b1 = ",parameters["b2"])

def forward_propagation(parameters,X):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    
    #return chache that have Z1,2 A1,A2 
    
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    chash = {
        "Z1":Z1,
        "Z2":Z2,
        "A1":A1,
        "A2":A2,
        }
    return A2 ,chash
#print(forward_propagation(parameters, X)[0].shape)
def compute_cost(A2,Y):
     m = Y.shape[1] # number of example
     logprobs = np.multiply(Y ,np.log(A2)) + np.multiply((1-Y), np.log(1-A2))
     cost = (-1/m) * np.sum(logprobs)
     return cost
#cost = compute_cost(forward_propagation(parameters, X)[0], Y)
#print(cost)


def backward_propagation(parameter,cache,X,Y):
    m = Y.shape[1]
    
    w1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]
    
    # calculate d, db1, dW2, db2. 
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2,A1.T)
    db2 = (1/m) *(np.sum(dZ2,axis=1,keepdims=True))
    dZ1 = np.dot(W2.T,dZ2) * (1 - np.power(A1,2))
    d = (1/m) *(np.dot(dZ1,X.T))
    db1 = (1/m) *(np.sum(dZ1, axis=1, keepdims=True))
    
    grads = {"d": d,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads
#show me shape of db1
#print(backward_propagation(parameters,forward_propagation(parameters, X)[1],X,Y)["db1"].shape)


def update_parameters(parameters, grads, learning_rate):
   
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    d = grads["d"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate * d
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1":W1 ,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def model(X, Y, n_h, learning_rate, num_iterations = 1000):
    n_y = layer_size(X, Y)[2]
    n_x = layer_size(X, Y)[0]
    #intialze parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    print(parameters)
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)
        
    return parameters



parameters = model(X, Y, 4, 1.02,num_iterations=1000)
'''
def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    
    return predictions

predictions = predict(parameters, X)
print("predictions mean = " + str(np.mean(predictions)))
'''










