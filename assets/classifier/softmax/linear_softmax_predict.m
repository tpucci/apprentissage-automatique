function [ y_pred ] = linear_softmax_predict( model, X )
% Use the trained weights of this linear classifier to predict labels for
% data points.
% 
% Inputs:
% - model : a trained model containing softmax weights.
% - X: N x D array of training data. Each row is a D-dimensional point.
% 
% Returns:
% - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
%   array of length N, and each element is an integer giving the predicted
%   class.
    y_pred = [];
    
%     ###########################################################################
%     # TODO:                                                                   #
%     # Implement this method. Store the predicted labels in y_pred.            #
%     ###########################################################################
    scores = model.W * X';
    [~,y_pred] = max(scores);
%     ###########################################################################
%     #                           END OF YOUR CODE                              #
%     ###########################################################################
end

