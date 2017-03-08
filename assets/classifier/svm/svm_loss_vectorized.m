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

num_classes = size(W,1);

scores = W * X';
correct_indexes = (0:num_train-1)*num_classes+double(y');
correct_class_score = ones(num_classes,1)*scores(correct_indexes);
L = scores - correct_class_score + 1; % delta = 1

L(L<0) = 0;
loss = sum(sum(L))- 1*num_train; % On retire les Lyi comptés en trop
loss = loss/num_train;

% Regularization
loss = loss + 0.5 * reg * sum(sum((W.*W)));

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

L(L>0) = 1;
L(correct_indexes) = -sum(L);
dW=L*X;

dW = dW/num_train;
% Add regularization to the gradient
dW = dW + reg * W;

% #############################################################################
% #                             END OF YOUR CODE                              #
% #############################################################################

end

