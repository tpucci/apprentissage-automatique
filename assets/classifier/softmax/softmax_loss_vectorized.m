function [ loss, dW ] = svm_loss_vectorized( W, X, y, reg )
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
% # Compute the softmax loss and its gradient using no explicit loops.  #
% # Store the loss in loss and the gradient in dW. If you are not careful     #
% # here, it is easy to run into numeric instability. Don't forget the        #
% # regularization!                                                           #
% #############################################################################
      
  num_class = size(W,1);
  num_train = size(X,1);
  
  scores = W*X';
  scores = scores - ones(num_class,1)*max(scores);
  correct_indexes = (0:num_train-1)*num_class+double(y');

  sum_exp = sum(exp(scores));
  loss = sum(- log(exp(scores(correct_indexes)./sum_exp)));

  L = exp(scores)./(ones(num_class,1)*sum_exp);
  L(correct_indexes) = L(correct_indexes)-1;

  dW=L*X;

  % Right now the loss is a sum over all training examples, but we want it
  % to be an average instead so we divide by num_train
  loss = loss/num_train;

  % Average gradients as well
  dW = dW/num_train;

  % Add regularization to the loss.
  loss = loss + 0.5 * reg * sum(sum((W.*W)));

  % Add regularization to the gradient
  % your code
  dW = dW + reg * W;

  
% #############################################################################
% #                          END OF YOUR CODE                                 #
% #############################################################################
end

