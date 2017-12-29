function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m,1) X];  %add bias column of 1's to X
Theta1 = Theta1';   %Change theta from a m x n to a n x m matrix
Theta2 = Theta2';   %Change theta from a m x n to a n x m matrix
a1 = X*Theta1;      %Compute activation of layers 2
a1z = sigmoid(a1);  %Compute sigmoid of activation g(a1);
a1z = [ones(m,1) a1z];  %add bias column of 1's to a1z
a2 = a1z*Theta2;    %Compute activation of layer 3
a2z = sigmoid(a2);  %Compute sigmoid g(a2);

[max_value , ind] = max(a2z, [], 2); %find which the maximum value in each column and which column it's in
ind(ind==10) = 0;   %set all values of 10 = to value zero since numbers range from 0 to 9 not 10
p = ind;            %output column containing highest volumn, this column # corresponds to the number prediction i.e column 5 = prediction of 5









% =========================================================================


end
