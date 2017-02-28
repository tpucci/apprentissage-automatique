function [y_pred] = knn_predict_labels(model, dists, k)
%     Given a matrix of distances between test points and training points,
%     predict a label for each test point.
% 
%     Inputs:
%     - model: KNN model struct, it has two members:
%         model.X_train : A matrix of shape (num_train, D) containing train data.
%         model.y_train : A matrix of shape (num_train, 1) containing train labels.
%     - dists: A matrix of shape (num_test, num_train) where dists[i, j]
%       gives the distance betwen the ith test point and the jth training point.
%     - k : number of nearest neighbors
%     Returns:
%     - y: A matrix of shape (num_test,) containing predicted labels for the
%       test data, where y[i] is the predicted label for the test point X[i].  


%     #########################################################################
%     # TODO:                                                                 #
%     # Use the distance matrix to find the k nearest neighbors of the ith    #
%     # training point, and use model.train_labels to find the labels of these      #
%     # neighbors. Store these labels in closest_y.                           #
%     # Hint: Look up the function sort                             #
%     #########################################################################

        closest_y = zeros(size(dists,2),k);
        for i = 1:size(dists,1)
                dist_i = dist(i,:);
                [dist_i_sorted,indexes] = sort(dist_i);
                k_first_indexes = indexes(1:k);
                closest_y(i,:) = model.y_train(k_first_indexes,2);
        end

% 
%     #########################################################################
%     # TODO:                                                                 #
%     # Now that you have found the labels of the k nearest neighbors, you    #
%     # need to find the most common label in the list closest_y of labels.   #
%     # Store this label in y_pred[i]. Break ties by choosing the smaller     #
%     # label.        
%     # Hint: Look up the function mode
%     #########################################################################
    
       your code
% 
%     #########################################################################
%     #                           END OF YOUR CODE                            # 
%     #########################################################################
end