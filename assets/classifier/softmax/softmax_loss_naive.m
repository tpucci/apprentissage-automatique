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
  
  for i = 1:num_train
      scores = W*X(i, :)';
      scores = scores - max(scores);

      sum_exp = sum(exp(scores));
      loss = loss -log(exp(scores(y(i)))/sum_exp);

      for j = 1:num_class
      
        dW(j,:) = dW(j,:) + exp(scores(j))/sum_exp*X(i, :);

        if j == y(i)
          dW(j,:) = dW(j,:) - X(i, :);
        end

      end
  end

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

