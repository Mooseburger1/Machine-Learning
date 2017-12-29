function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


m = 8; 
running_error = eye(m); %initialize vector to contain all error values from each test 8 C values tested with 8 sigma values 8^2 = 64 different error values
i = 0;

for C_try = [0.01 , 0.03 , 0.1 , 0.3 , 1 ,3 ,10 , 30] %different values of C to test
  i = i+1;
  j = 1;
  for sigma_try = [0.01 , 0.03 , 0.1 , 0.3 , 1 , 3 , 10 , 30] %different values of sigma to test
    model= svmTrain(X, y, C_try, @(x1, x2) gaussianKernel(x1, x2, sigma_try));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    running_error(i,j) = error;
    j = j + 1;
  end
end


[minError,ind] = min(running_error(:)); % find the minimum error value from the tests
[m,n] = ind2sub(size(running_error),ind); % find the index of that minimum value

sigma_try = [0.01 , 0.03 , 0.1 , 0.3 , 1 , 3 , 10 , 30];
C_try = [0.01 , 0.03 , 0.1 , 0.3 , 1 ,3 ,10 , 30];

C = C_try(m);
sigma = sigma_try(n);
 
    




% =========================================================================

end
