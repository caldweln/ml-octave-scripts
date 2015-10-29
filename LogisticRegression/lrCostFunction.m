function [J, grad] = lrCostFunction(theta, X, y, lambda)
%   [J, grad] = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

	% default lambda to zero, ignoring regulariation
	if(nargin < 4)
		lambda = 0;
	endif

	m = length(y); % number of training examples
	grad = zeros(size(theta));

	% computing cost J

	h = sigmoid( X * theta);
	pos_cost = y' * log(h);
	neg_cost = ( 1 .- y)' * log( 1 .- h);

	J = (-1/m) * ( pos_cost + neg_cost );

	% adding regularization to cost J
	theta_unbias = theta;
	theta_unbias(1) = 0;

	reg_cost = (lambda/(2*m)) * theta_unbias'*theta_unbias;
	J = J + reg_cost;

	% computing gradient grad of regularized logistic regression cost

	err = h - y;
	reg_grad = lambda * (1/m) .* theta_unbias;
	grad = (1/m) * X' * err + reg_grad;

	grad = grad(:);

end
