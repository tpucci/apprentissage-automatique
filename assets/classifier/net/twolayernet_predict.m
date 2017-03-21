function [ y_pred ] = twolayernet_predict( model, X )
%  Use the trained weights of this two-layer network to predict labels for
%     data points. For each data point we predict scores for each of the C
%     classes, and assign each data point to the class with the highest score.
% 
%     Inputs:
%     - model : a struct having weights of network;
%     - X: A matrix of shape (N, D) giving N D-dimensional data points to
%       classify.
% 
%     Returns:
%     - y_pred: A matrix of shape (N,1) giving predicted labels for each of
%       the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
%       to have class c, where 1 <= c <= C.

    y_pred = [];
      
%     ###########################################################################
%     # TODO: Implement this function; it should be VERY simple!                #
%     ###########################################################################
      scores = twolayernet_loss( model, X);
      [~,y_pred] = max(scores,[],2); 
      y_pred = y_pred'; 
      
%     ###########################################################################
%     #                              END OF YOUR CODE                           #
%     ###########################################################################

end

