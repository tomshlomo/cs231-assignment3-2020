from builtins import range
import numpy as np
from random import shuffle
#from past.builtins import xrange

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
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    C = W.shape[1]
    f = np.dot(X,W) # N x C
    for i in range(N):
        m = np.max(f[i,:])
        f[i,:] -= m
        exp_vec = np.exp( f[i,:] )
        sum_exp = np.sum( exp_vec )
        loss += -f[i,y[i]] + np.log( sum_exp )
        
        dW[:,y[i]] -= X[i,:]
        for j in range(C):
            dW[:,j] += ( exp_vec[j]*X[i,:] )/sum_exp
        # dL/dW[:,j] = ___ + 1/(sum_exp) * { exp_vec[j]* X[i,:] }
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss /= N
    dW /= N
    loss += reg*np.sum(W*W)
    
    dW += reg*2*W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    C = W.shape[1]
    D = W.shape[0]
    f = np.dot(X,W) # N x C
    
    f -= np.max(f,axis=1).reshape([-1,1])
    f_y = f[np.arange(N),y]
    loss = np.mean( -f_y + np.log( np.sum( np.exp(f), axis=1 ) ) )
    
    M = np.zeros([C,N])
    M[y,np.arange(N)] = -1
    dW = np.dot(M,X).transpose()
    P = np.exp(f)
    P /= np.sum(P, axis=1).reshape([-1,1]) # N x C
    dW += np.dot(X.transpose(), P)
    dW /= N
    
    loss += reg*np.sum(W*W)
    dW += reg*2*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
