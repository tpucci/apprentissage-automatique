function [ loss, grads ] = twolayernet_loss( model, X, y, reg )
% Compute the loss and gradients for a two layer fully connected neural
%     network.
% 
%     Inputs:
%     - model : a struct containing network wegihts;
%     - X: Input data of shape (N, D). Each X[i] is a training sample.
%     - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
%       an integer in the range 1 <= y[i] <= C. This parameter is optional; if it
%       is not passed then we only return scores, and if it is passed then we
%       instead return the loss and gradients.
%     - reg: Regularization strength.
% 
%     Returns:
%     If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
%     the score for class c on input X[i].
% % 
%     If y is not None, instead return a tuple of:
%     - loss: Loss (data loss and regularization loss) for this batch of training
%       samples.
%     - grads: Dictionary mapping parameter names to gradients of those parameters
%       with respect to the loss function; has the same keys as self.params.
    if nargin < 4
        reg = 0.0;
    end
    
    [N, D] = size(X);
    % Compute the forward pass
    scores = 0;
    
%   #############################################################################
%   # TODO: Perform the forward pass, computing the class scores for the input. #
%   # Store the result in the scores variable, which should be an array of      #
%   # shape (N, C).            
%   # Hint: input - fully connected layer - ReLU - fully connected layer
%   #############################################################################
    res=X * model.W1 + repmat(model.b1,N,1);
    hidden_layer = max(0,res);
    scores =hidden_layer * model.W2 + repmat(model.b2,N,1); 
    
%   #############################################################################
%   #                              END OF YOUR CODE                             #
%   #############################################################################
    if (nargin == 2) || (nargin==3 && isscalar(y))
        loss = scores;
        return;
    end
    
    % Compute the loss
    loss = 0;
%   #############################################################################
%   # TODO: Finish the forward pass, and compute the loss. This should include  #
%   # both the data loss and L2 regularization for W1 and W2. Store the result  #
%   # in the variable loss, which should be a scalar. Use the Softmax           #
%   # classifier loss. So that your results match ours, multiply the            #
%   # regularization loss by 0.5                                                #
%   #############################################################################
    scores = scores - repmat(max(scores,[],2),1,size(scores,2));

    scores = exp(scores) ./ repmat(sum(exp(scores),2),1,size(scores,2)); 
	true_class = false(size(scores));
	true_class(sub2ind(size(scores),1:N,y')) = 1; 
	loss = - sum(sum(true_class .* log(scores)));
    
    loss = loss ./ N;
    loss = loss + 0.5 * reg * (sum(sum((model.W1.*model.W1))) + sum(sum((model.W2.*model.W2))));
	
%   #############################################################################
%   #                              END OF YOUR CODE                             #
%   #############################################################################


%   Backward pass: compute gradients
    grads = {};
%   #############################################################################
%   # TODO: Compute the backward pass, computing the derivatives of the weights #
%   # and biases. Store the results in the grads struct. For example,       #
%   # grads.W1 should store the gradient on W1, and be a matrix of same size #
%   #############################################################################
    dout2 = (scores - true_class) ./ N;
    dW2 = hidden_layer' * dout2;
    db2 = ones(N,1)' * dout2;
    
    dout1 = dout2 * model.W2';
    dout1(res < 0) = 0;
    
    dW1 = X' * dout1;
    db1 = ones(N,1)' * dout1;
    
    dW2 = dW2 + reg * model.W2;
    dW1 = dW1 + reg * model.W1;
    
    grads.W1 = dW1;
    grads.b1 = db1;
    grads.W2 = dW2;
    grads.b2 = db2;
    	
    
%   #############################################################################
%   #                              END OF YOUR CODE                             #
%   #############################################################################
end

