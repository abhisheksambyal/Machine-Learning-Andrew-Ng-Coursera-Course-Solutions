function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m, 1) X];

a1 = X;
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3); %% a3 :- hypothesis
y_matrix = zeros(num_labels, m);

%{
for i = 1:m
  if(y(i) == 10)
    yy(10, i) = 1;
  else
    yy(y(i), i) = 1;
  endif
endfor
%disp(yy(:, 5000));
%}

%% Easy way in OCTAVE
y_matrix = eye(num_labels)(y,:);

%{
m = the number of training examples
n = the number of training features, including the initial bias unit.
h = the number of units in the hidden layer - NOT including the bias unit
r = the number of output classifications

1: Perform forward propagation, see the separate tutorial if necessary.
2: δ3 or d3 is the difference between a3 and the y_matrix. The dimensions are
   the same as both, (m x r).
3: z2 came from the forward propagation process - it's the product of a1 and
   Theta1, prior to applying the sigmoid() function. Dimensions are (m x n)
   ⋅ (n x h) --> (m x h)
4: δ2 or d2 is tricky. It uses the (:,2:end) columns of Theta2. d2 is the
   product of d3 and Theta2(no bias), then element-wise scaled by sigmoid
   gradient of z2. The size is (m x r) ⋅ (r x h) --> (m x h). The size is
   the same as z2, as must be.
5: Δ1 or Delta1 is the product of d2 and a1. The size is (h x m) ⋅ (m
   x n) --> (h x n)
6: Δ2 or Delta2 is the product of d3 and a2. The size is (r x m)
   ⋅ (m x [h+1]) --> (r x [h+1])
7: Theta1_grad and Theta2_grad are the same size as their
   respective Deltas, just scaled by 1/m.
Now you have the unregularized gradients. Check your results
using ex4.m, and submit this portion to the grader.
%}

Delta1 = 0;
Delta2 = 0;
d3 = a3 - y_matrix;
d2 = d3 * Theta2(:, 2:end) .* sigmoidGradient(z2);
Delta1 = Delta1 + d2' * a1;
Delta2 = Delta2 + d3' * a2;

%{
===== Regularization of the gradient ===========

Since Theta1 and Theta2 are local copies, and we've already computed our
hypothesis value during forward-propagation, we're free to modify them to make
the gradient regularization easy to compute.

8: So, set the first column of Theta1 and Theta2 to all-zeros.
9: Scale each Theta matrix by λ/m. Use enough parenthesis so the operation
   is correct.
10: Add each of these modified-and-scaled Theta matrices to the
    un-regularized Theta gradients that you computed earlier.
%}

Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;

t1 = Theta1;
t2 = Theta2;
t1(:,1) = 0;
t2(:,1) = 0;

t1 = (t1 * lambda) / m;
t2 = (t2 * lambda) / m;

Theta1_grad = Theta1_grad + t1;
Theta2_grad = Theta2_grad + t2;

%% Iterative version for calculating cost
%{
for i = 1:m
  for k = 1:num_labels
    J += (-y_matrix(i, k) * log(hyp(i, k))) - ((1 - y_matrix(i, k)) * log(1 - hyp(i, k)));
  endfor
endfor
J = J / m;
%}

% Vectorized version
J = sum(sum((-y_matrix .* log(a3)) - ((1 - y_matrix) .* log(1 - a3)))) / m;

reg_theta1 = sum(sum(Theta1(:, 2:end) .^ 2));
reg_theta2 = sum(sum(Theta2(:, 2:end) .^ 2));
reg_t = reg_theta1 + reg_theta2;

J = J + ((lambda * reg_t) / (2 * m));







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
