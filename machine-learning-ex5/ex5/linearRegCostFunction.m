function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = X*theta;
theta_temp = theta;
theta_temp(1) = 0;

%Compute cost function - rember not to regularize theta1 bias unit
J = (1/(2.*m)) .* (sum((h.-y).^2)) .+ ((lambda/(2*m)).*(sum(theta_temp.^2)));

%Compute gradient - remember to transpose first part of the addition to make args match
grad = ((1/m) .* (sum((h.-y).*X)))' .+ ((lambda/m)*theta_temp);










% =========================================================================

grad = grad(:);

end
