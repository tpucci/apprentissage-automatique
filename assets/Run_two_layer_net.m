function [ output_args ] = Run_two_layers_net( input_args )
%Two layer net exercise

addpath('./classifier/net');
addpath('./dataset');

%% We will use the function `twolayernet_init` in the folder `./classifier/net/
% ` to initialize instances of our network. The network parameters are stored
% in the struct `model` where values are matrices. Below, we initialize toy
% data and a toy model that we will use to develop your implementation.

% Create a small net and some toy data to check your implementations.
% Note that we set the random seed for repeatable experiments.

input_size = 4;
hidden_size = 10;
num_classes = 3;
num_inputs = 5;

function [model] = init_toy_model()
    rng(0);
    model = twolayernet_init(input_size, hidden_size, num_classes, 1e-1);
end

function [X, y] = init_toy_data()
    rng(1);    
    X = 10 * randn(num_inputs, input_size);
    y = [0; 1; 2; 2; 1] + 1;
end

model = init_toy_model();
[X,y] = init_toy_data();
%% Forward pass: compute scores
% Open the file './classifiers/net/twolayernet_loss.m`. This function is
% very similar to the loss functions you have written for the SVM and 
% Softmax exercises: It takes the data and weights and computes the class 
% scores, the loss, and the gradients on the parameters.

% Implement the first part of the forward pass which uses the weights and 
% biases to compute the scores for all inputs.
[scores] = twolayernet_loss(model,X);
disp('Your scores:');
disp(scores);
correct_scores = [0.39674245, -0.13838192, -0.48429209;
                  0.85662955,  0.42995926,  0.98941754;
                  0.33614122, -0.26699700, -0.59185479;
                 -0.19349664,  0.29995214,  0.39273830;
                  0.26726185, -0.67868934,  0.09010580];
disp('Correct scores:');
disp(correct_scores);             
%The difference should be very small. We get < 1e-7
fprintf('Difference between your scores and correct scores:\n');
fprintf('%e\n', sum(sum(abs(scores - correct_scores))));

%% Forward pass: compute loss
% In the same function, implement the second part that computes the data and regularizaion loss.
[loss, ~] = twolayernet_loss(model,X,y, 0.1);
correct_loss = 1.330037538281351;
% should be very small, we get < 1e-12
fprintf('Difference between your loss and correct loss:\n');
fprintf('%e\n', abs(loss - correct_loss));

%% Backward pass
%Implement the rest of the function. This will compute the gradient of the loss 
% with respect to the variables W1, b1, W2, and b2. Now that you (hopefully!)
% have a correctly implemented forward pass, you can debug your backward pass 
% using a numeric gradient check:
function [error] = rel_error(x, y)
% returns relative error """
    p = abs(x) + abs(y);
    p(p<1e-8) = 1e-8;
    error = max(max(abs(x-y)./p));
end
[loss, grad,] = twolayernet_loss(model, X, y, 0.1);
f =@(m)twolayernet_loss(m, X, y, 0.1);
fields = fieldnames(grad);
for i = 1:length(fields)
    [grad_eval] = eval_numerical_gradient(f, model, fields{i}, 0);
    fprintf('%s max relative error :%e\n', fields{i}, rel_error(grad_eval, grad.(fields{i})));
end

%% Train the network
%To train the network we will use stochastic gradient descent (SGD), 
%similar to the SVM and Softmax classifiers. Look at the function 
%`twoLayernet_train` and fill in the missing sections to implement the
%training procedure. This should be very similar to the training procedure
%you used for the SVM and Softmax classifiers. You will also have to implement 
%`twoLayernet_predict`, as the training process periodically performs prediction 
%to keep track of accuracy over time while the network trains.

%Once you have implemented the method, run the code below to train a two-layer 
%network on toy data. You should achieve a training loss less than 0.3.
model = init_toy_model();
params.learning_rate = 1e-1;
params.reg = 1e-5;
params.num_iters = 100;
params.verbose = 0;
[model, stats] = twolayernet_train(model,X,y,X,y,params);

fprintf('Final training loss: %f\n', stats.loss_history(end));
% plot the loss history
figure;
plot(stats.loss_history);
xlabel('iteration');
ylabel('training loss');
title('Training Loss history');

%% Load the data
% Now that you have implemented a two-layer network that passes gradient checks 
% and works on toy data, it's time to load up our favorite CIFAR-10 data so 
% we can use it to train a classifier on a real dataset.

imdb = prepare_net_datasets();
% As a sanity check, we print out the size of the training and test data.
disp('Training data shape: ');
disp(size(imdb.X_train));
disp('Training labels shape: ');
disp(size(imdb.y_train));
disp('Validation data shape: ');
disp(size(imdb.X_val));
disp('Validation labels shape: ');
disp(size(imdb.y_val));
disp('Test data shape: ');
disp(size(imdb.X_test));
disp('Test labels shape: ');
disp(size(imdb.y_test));
disp('==========================================');

%% Train a network
% To train our network we will use SGD with momentum. In addition, we will 
% adjust the learning rate with an exponential learning rate schedule as 
% optimization proceeds; after each epoch, we will reduce the learning rate
% by multiplying it by a decay rate

input_size = 32 * 32 * 3;
hidden_size = 50;
num_classes = 10;

model = twolayernet_init(input_size, hidden_size, num_classes);
params.num_iters = 1000;
params.batch_size = 200;
params.learning_rate = 1e-4;
params.learning_rate_decay = 0.95;
params.reg = 0.5;
params.verbose = 1;
[model, stats] = twolayernet_train(model, imdb.X_train, imdb.y_train, ...
                                   imdb.X_val, imdb.y_val, params);
%Predict on the validation set
val_acc = mean(twolayernet_predict(model, imdb.X_val) == imdb.y_val');
fprintf('Validation accuracy: %f\n', val_acc);


end

