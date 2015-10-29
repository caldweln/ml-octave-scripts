function [J, grad] = costFunction(X, y, theta, lambda)
%   [J, grad] = COSTFUNTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

	% default lambda to zero, ignoring regulariation
	if(nargin < 4)
		lambda = 0;
	endif

	m = length(y); % number of training examples
	grad = zeros(size(theta));


	% computing cost J

	h = X*theta;
	cost = h-y;
	J = (1/(2*m))*(cost'*cost);

	% adding regularization to cost J

	theta_unbias = theta;
	theta_unbias(1,:) = 0;

	reg_cost = (lambda/(2*m))*(theta_unbias'*theta_unbias);
	J = J + reg_cost;

	% computing gradient grad of regularized linear regression cost

	reg_grad = (lambda/m)*theta_unbias;
	grad = (1/m)*(X'*cost) + reg_grad;

	grad = grad(:);

end
