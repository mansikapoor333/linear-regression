"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #print(w.shape, X.shape)
    y_prediction=predict(X,w)
    #print(y_prediction)
    assert len(y)==len(y_prediction)
    #print("MAE", y.shape, y_prediction.shape)
    return np.mean(np.abs(y-y_prediction))
    #err = None
    #return err

def predict(features, weights):
    n=len(features)
    #print(n)
    #x = np.hstack((np.ones((n, 1)), np.array(features)))
    #print(x)
    y=features.dot(weights)
    #print(y)
    return np.array(y).flatten()

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  
  #n = len(X)
  #x_noreg = np.hstack((np.ones((n, 1)), np.array(X)))
  y1 = np.array(y)
  weights = np.linalg.inv(X.T.dot(X))
  weights1=weights.dot(X.T)
  e=weights1.dot(y1)
  return np.array(e)
	
  #w = None
  #return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #X_T = np.transpose(X)
    #K=X_T.dot(X)
    #w,v = np.linalg.eig(K)
    #while not all(i >= 10**(-5) for i in w):
        #K=K+(0.1*np.identity(len(K)))
        #w, v = np.linalg.eig(K)
        
    #y1 = np.array(y)
    #weights = np.linalg.pinv(K).dot(X.T).dot(y1)
    #return np.array(weights)
    def getEigen(z):
        return np.linalg.eig(z)

    X_T = np.transpose(X)
    K = X_T.dot(X)
    wt, vec = np.linalg.eig(K)
    minimum = -100000
    while minimum < 10 ** (-5):
        K = K + (0.1 * np.identity(len(K)))
        w, v = getEigen(K)
        minimum = np.amin(np.abs(w))
    y1 = np.array(y)
    weights = np.linalg.pinv(K)
    weights1 = weights.dot(X.T)
    weights2 = weights1.dot(y1)
    return np.array(weights2)
    
                      
    
    
    #w = None
    #return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  
    n = len(X)
    p = len(X[0])
    diag = lambd * np.identity(p)
    weights = np.linalg.inv(X.T.dot(X) + diag)
    weights1=weights.dot(X.T)
    weights3=weights1.dot(y)
    return np.array(weights3)	
    #w = None
    #return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    best_mae, best_lambda=1e10, -1
    for i in range(-19,20,1):
        w= regularized_linear_regression(Xtrain, ytrain,10**(i))
        #print(w)
        train_mae= mean_absolute_error(w,Xtrain, ytrain)
        #print train_mae
        valid_mae= mean_absolute_error(w,Xval, yval)
        #print valid_mae
        if valid_mae < best_mae:
            best_mae, bestlambda = valid_mae, (10**i)
    return bestlambda
        
    		
    #bestlambda = None
    #return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    k=X
    for i in range(2,power+1):
        x1=np.power(k,i)
        X=np.concatenate((X,x1),axis=1)
    return X
    


