function [ loss, dW  ] = svm_loss_vectorized( W, X, y, reg )
% """
% Structured SVM loss function, vectorized implementation.
% Inputs and outputs are the same as svm_loss_naive.
% Inputs:
% - W: C x D matrix of weights
% - X: N x D matrix of data. Data are D-dimensional columns
% - y: 1-dimensional array of length N with labels 0...K-1, for K classes
% - reg: (float) regularization strength
% Returns:
% a tuple of:
% - loss as single float
% - gradient with respect to weights W; an array of same shape as W
% """

loss = 0.0;
dW = zeros(size(W)); % initialize the gradient as zero
num_train = size(X, 1);

% #############################################################################
% # TODO:                                                                     #
% # Implement a vectorized version of the structured SVM loss, storing the    #
% # result in loss.                                                           #
% #############################################################################

your code 

% #############################################################################
% #                             END OF YOUR CODE                              #
% #############################################################################

% #############################################################################
% # TODO:                                                                     #
% # Implement a vectorized version of the gradient for the structured SVM     #
% # loss, storing the result in dW.                                           #
% #                                                                           #
% # Hint: Instead of computing the gradient from scratch, it may be easier    #
% # to reuse some of the intermediate values that you used to compute the     #
% # loss.                                                                     #
% #############################################################################

your code 

% #############################################################################
% #                             END OF YOUR CODE                              #
% #############################################################################

end

