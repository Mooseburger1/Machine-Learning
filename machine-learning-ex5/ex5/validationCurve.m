function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%
[m n] = size(X);

for i = 1:m
for j = 1:length(lambda_vec)

lambda = lambda_vec(j)

%makes a subset of training examples from main training set
X_train = X(1:i , :);
y_train = y(1:i);

%calculate optimum theta for subset training set for particular loop
theta = trainLinearReg(X_train,y_train,lambda);

%calculate cost and gradient of subset training data with calc optimum theta
[train_cost , train_grad] = linearRegCostFunction(X_train,y_train,theta,0);
%we dont loop through the cross validation like the training set because we are inputting and comparing the error of our val set with the subset of training set of m values. 
[val_cost , val_grad] = linearRegCostFunction(Xval,yval,theta,0);

%update the error for each subset to plot #of training examples vs error
error_train(j) = train_cost;
error_val(j) = val_cost;




end

% =========================================================================

end
