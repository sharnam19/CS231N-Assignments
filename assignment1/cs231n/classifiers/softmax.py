import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_train = X.shape[0]
  num_class = W.shape[1]
  dW= np.zeros_like(W)
  for i in range(num_train):
        score_i = X[i].dot(W)
        score_i -= np.max(score_i)
        exp_score_i = np.exp(score_i)
        p_i = exp_score_i/np.sum(exp_score_i)
        loss+=-np.log(p_i[y[i]])
        for j in range(num_class):
            dW[:,j] += (p_i[j]-(j==y[i]))*X[i]
  dW/=num_train
  dW+=2*reg*W
  loss/=num_train
  loss += reg*np.sum(np.square(W))
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_train =X.shape[0]
  dW = np.zeros_like(W)
  score = X.dot(W)
  score -= np.max(score,axis=1).reshape(-1,1)
  exp_score = np.exp(score)
  p_score = exp_score /np.sum(exp_score,axis=1).reshape(-1,1)
  loss = np.sum(-np.log(p_score[range(num_train),y]))/num_train + reg*np.sum(np.square(W))
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  trueV=np.zeros_like(p_score)
  trueV[range(num_train),y]=1
  dW = X.T.dot(p_score-trueV)/num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW

