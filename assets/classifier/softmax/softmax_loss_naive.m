function [ loss, dW ] = softmax_loss_naive( W, X, y, reg )
% Softmax loss function, naive implementation (with loops)
% Inputs:
% - W: C x D matrix of weights
% - X: N x D matrix of data. Data are D-dimensional columns
% - y: 1-dimensional array of length N with labels 1..k , for K classes
% - reg: (float) regularization strength
% Returns:
% a tuple of:
% - loss 
% - gradient with respect to weights W, an array of same size as W
  
% Initialize the loss and gradient to zero.
  loss = 0.0;
  dW = zeros(size(W));

% #############################################################################
% # TODO: Compute the softmax loss and its gradient using explicit loops.     #
% # Store the loss in loss and the gradient in dW. If you are not careful     #
% # here, it is easy to run into numeric instability. Don't forget the        #
% # regularization!                                                           #
% #############################################################################
  
  num_class = size(W,1);
  num_train = size(X,1);
  
  

  
% #############################################################################
% #                          END OF YOUR CODE                                 #
% #############################################################################
end

